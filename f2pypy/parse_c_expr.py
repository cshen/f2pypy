# pyf files contain embedded C code.

# Some examples are:
#  integer optional, intent(in),check(incx>0||incx<0) :: incx = 1
#  integer optional,intent(in),depend(x,incx,offx,y,incy,offy) :: n = (len(x)-offx)/abs(incx)
#  integer intent(hide),depend(incy,rows,offy) :: ly = (y_capi==Py_None?1+offy+(rows-1)*abs(incy):-1)
#    (I haven't yet figured out how to handle the one above; I kinda faked it.)
#  callstatement (*f2py_func)((trans?(trans==2?"C":"T"):"N"),&m,&n,&alpha,a,&m,x+offx,&incx,&beta,y+offy,&incy)


# I parse those expressions based on the code by Fredrik Lundh, at
#   http://effbot.org/zone/simple-top-down-parsing.htm

# The result is an AST. I do several passes over the AST to make a new
# AST which I can then convert into valid Python code.

# Scalar values are turned into c_types variables. (Should I use
# scalar numpy arrays instead?) Vector values are turned into numpy
# arrays.

# This code expects a certain run-time environment, which is
# incomplete. For example, this does not implement a version of the
# array creation function which f2py uses. Instead, it uses
# numpy.array, which doesn't have the same copy semantics.

# Instead, this code is more of a proof-of-concept that translation
# *is* possible.


# Note: this is single threaded

import sys
import re
from cStringIO import StringIO

symbol_table = {}

class symbol_base(object):
    value = None
    def __repr__(self):
        if self.value is not None:
            return "%s %r)" % (self.id[:-1], self.value)
            #return "(literal %s)" % self.value
        elif hasattr(self, "third"):
            return "(%s %s %s %s)" % (self.id, self.first, self.second, self.third)
        else:
            return "(%s %s %s)" % (self.id, self.first, self.second)

def symbol(id, bp=0):
    try:
        s = symbol_table[id]
    except KeyError:
        class s(symbol_base):
            pass
        s.__name__ = "symbol-" + id # for debugging
        s.id = id
        s.value = None
        s.lbp = bp
        symbol_table[id] = s
    else:
        s.lbp = max(bp, s.lbp)
    return s

def advance(id=None):
    global token
    if id and token.id != id:
        raise SyntaxError("Expected %r" % id)
    token = next()

def paren_nud(self):
    expr = expression()
    # TODO: Check for empty parens()
    advance(")")
    return expr

def paren_led(self, left): # for use as a function cal
    self.first = left
    self.second = expression()
    advance(")")
    return self

symbol("(").nud = paren_nud
symbol("(").led = paren_led
symbol(")")


def led(self, left):
    self.first = left
    self.second = expression()
    advance(":")
    self.third = expression()
    return self
symbol("?").led = led
symbol(":")
    

# manual configuration

def led(self, left):
    self.first = left
    self.second = expression(10)
    return self
symbol("+").led = led
symbol("-").led = led

# helpers

def infix(id, bp):
    def led(self, left):
        self.first = left
        self.second = expression(bp)
        return self
    symbol(id, bp).led = led

def infix_r(id, bp):
    def led(self, left):
        self.first = left
        self.second = expression(bp-1)
        return self
    symbol(id, bp).led = led

def prefix(id, bp):
    def nud(self):
        self.first = expression(bp)
        self.second = None
        return self
    symbol(id).nud = nud


infix("=", 10)
infix(",", 15)
symbol("?", 20)
infix_r("||", 30); infix_r("&&", 40); prefix("!", 50)

infix("<", 60); infix("<=", 60);
infix(">", 60); infix(">=", 60);
infix("==", 60)

# infix("|", 70); infix("^", 80);

# infix("<<", 100); infix(">>", 100)

infix("+", 110); infix("-", 110)
infix("*", 120); infix("/", 120);
infix("%", 120)

prefix("-", 130); prefix("+", 130); prefix("~", 130)
prefix("&", 130)

symbol(".", 150); symbol("[", 150); symbol("(", 150)


symbol("(symbol)")
symbol("(end)")

symbol("(number)").nud = lambda self: self
symbol("(string)").nud = lambda self: self
symbol("(symbol)").nud = lambda self: self

# The "hack!" is the hack to support complex numbers.
# I don't know what f2py does. I make it be a special token.
# But this means that f(1,2) will look like a tuple! XXX

token_pat = re.compile(r"""
\s*(
(?P<number>\d+(\.\d+)?) |
(?P<string>\"[^"]*") |
(?P<complex>\((?P<_real>[+-]?\d+(\.\d+)?)\s*,\s*(?P<_imag>[+-]?\d+(\.\d+)?)) | # hack!
(?P<symbol>[a-zA-Z_][a-zA-Z0-9_]*) |
(?P<operator>\*\*|[+*/?:-]|==|=|\|\||\&\&|\(|\)|>=|<=|==|[<>=&|,])
)""", re.X)

def tokenize(program):
    pos = 0
    for m in token_pat.finditer(program):
        if m is None or m.start() != pos:
            raise SyntaxError("Unknown syntax at position %d in %r" % (pos, program))
        d = m.groupdict()
        for (name, tok) in d.items():
            if tok is None or name[:1] == "_":
                continue
            if name == "number":
                symbol_class = symbol_table["(number)"]
                symbol = symbol_class()
                symbol.value = tok
                break

            if name == "string":
                symbol_class = symbol_table["(string)"]
                symbol = symbol_class()
                symbol.value = tok
                break

            if name == "complex":
                symbol_class = symbol_table["(number)"]
                symbol = symbol_class()
                real = m.group("_real")
                imag = m.group("_imag")
                if imag[:1] not in "+-":
                    imag = "+" + imag
                symbol.value = "(%s%sj)" % (real, imag)
                break

            if name == "symbol":
                symbol_class = symbol_table["(symbol)"]
                symbol = symbol_class()
                symbol.value = tok
                break

            if name == "operator":
                symbol_class = symbol_table.get(tok)
                if not symbol_class:
                    raise SyntaxError("Unknown operator: %r" % (tok,))
                symbol = symbol_class()
                break
            
            raise AssertionError("Bad tokenizer")
        else:
            raise AssertionError("Bad tokenizer")

        yield symbol
        pos = m.end()

    if pos != len(program):
        s = program[pos:]
        if not s.isspace():
            raise AssertionError("Bad %d: %r" % (pos, s))
    
    symbol = symbol_table["(end)"]
    yield symbol()

def expression(rbp=0):
    global token
    t = token
    token = next()
    left = t.nud()
    while rbp < token.lbp:
        t = token
        token = next()
        left = t.led(left)
    return left

def parse(program):
    global token, next
    next = tokenize(program).next
    token = next()
    return expression()


################## Code to convert an AST into Python code

class EmitPythonExpression(object):
    # Map from token id to handler (internal class variable)
    emit_handlers = {}

    # Helper decorator to add handlers
    def match(symbol, needs_parens=True, extra=None, emit_handlers=emit_handlers):
        def decorator(func):
            emit_handlers[symbol] = (func, needs_parens, extra)
        return decorator

    # Write (unmodified) text to the output stream
    def write(self, text):
        self.outfile.write(text)

    # Save the converted AST to a given file
    def emit_to_file(self, ast, outfile):
        self.outfile = outfile
        self.depth = 0
        self._emit(ast)
        self.outfile = None

    # Convert the AST into a string of Python code
    def emit_to_string(self, ast):
        outfile = StringIO()
        self.emit_to_file(ast, outfile)
        return outfile.getvalue()
        
    # Internal function to descend into the parse tree
    def _emit(self, ast):
        func, needs_parens, extra = self.emit_handlers[ast.id]
        self.depth += 1
        try:
            if needs_parens:
                self.write("(")
                func(self, ast, extra)
                self.write(")")
            else:
                func(self, ast, extra)
        finally:
            self.depth -= 1

    # Helper function for binary operators which
    # have the same name in C and Python
    def convert_unchanged(self, ast, extra):
        self._emit(ast.first)
        self.write(" %s " % (ast.id,))
        self._emit(ast.second)

    # Helper function for binary operators which
    # have a differetn name in C and Python
    def convert_rename(self, ast, rename):
        self._emit(ast.first)
        self.write(" %s " % (rename,))
        self._emit(ast.second)

    # Helper function for operators which can be unary or binary
    def convert_unary_or_binary(self, ast, symbol):
        if ast.second is None:
            self.write(symbol)
            self._emit(ast.first)
        else:
            self._emit(ast.first)
            self.write(symbol)
            self._emit(ast.second)


    # These all fit into the same binary operation category
    for symbol in ("< <= == >= > * % /".split()):
        emit_handlers[symbol] = (convert_unchanged, True, None)

    # These are a bit special because they can be unary
    for symbol in ("+ -".split()):
        emit_handlers[symbol] = (convert_unary_or_binary, True, symbol)

    # These have different names in Python.
    for symbol, rename in (("&&", "and"), ("||", "or")):
        emit_handlers[symbol] = (convert_rename, True, rename)


    @match("(symbol)", False)
    def convert_symbol(self, ast, extra):
        if ast.value.lower() == "py_none": # f2py lowercases its variables
            self.write("None")
        elif ast.value.endswith("_capi"):
            # XXX I'm not sure about this! It looks like you can include
            # checks based on what the PyObject* was. I don't support
            # this feature, so I'll fake it for now.
            self.write(ast.value[:-5])
        else:
            self.write(ast.value)

    @match("(number)", False)
    def convert_number(self, ast, extra):
        self.write(ast.value)

    @match("(string)", False)
    def convert_string(self, ast, extra):
        # TODO: better support for string regex
        # TODO: convert from C string to Python string.
        self.write(ast.value)

    @match("=", False)
    def convert_assignment(self, ast, extra):
        assert self.depth == 1
        assert ast.first.id == "(symbol)", ast.first.id
        self._emit(ast.first)
        self.write(" = ")
        self._emit(ast.second)
        
    @match("(", False)
    def convert_function_call(self, ast, extra):
        # The special intrinsics are "len" and "shape". Are there others?
        # ("abs" isn't treated specially.)
        if ast.first.value == "len":
            self.write("(")
            self._emit(ast.second)
            self.write(").size")
        elif ast.first.value == "shape":
            self.write("_shape(")
            self.write(ast.second.first.value)
            self.write(", ")
            self._emit(ast.second.second)
            self.write(")")
        else:
            self._emit(ast.first)
            self.write("(")
            self._emit(ast.second)
            self.write(")")

    @match("?", needs_parens=False)
    def convert_ternary(self, ast, extra):
        self.write("(")
        self._emit(ast.second)
        self.write(")")
        self.write(" if ")
        self.write("(")
        self._emit(ast.first)
        self.write(")")
        self.write(" else ")
        self.write("(")
        self._emit(ast.third)
        self.write(")")

    @match(",", needs_parens=False)
    def convert_comma(self, ast, extra):
        self._emit(ast.first)
        self.write(", ")
        self._emit(ast.second)

    @match("&")
    def convert_deref(self, ast, extra):
        self.write(ast.first.value)
        #self._emit(ast.first)

    @match("(vector)", needs_parens=False)
    def convert_vector(self, vec, extra):
        if vec.first is None:
            self.write(vec.value)
        else:
            # This might be a 0 dimensional array
            self.write("(%s if (" % (vec.value,))
            self._emit(vec.first)
            self.write(") == 0 else %s[" % (vec.value,))
            self._emit(vec.first)
            self.write(":])")
        self.write(".ctypes.data_as(")
        self.write(vec.typeinfo)
        self.write(")")

    @match("(scalar)")
    def convert_vector(self, vec, extra):
        self.write(vec.value + ".value")
           
####### Code to transform the AST into a more appropriate form

# Helper function to do a depth-first descent of the AST and let a
# callback function modify the nodes in-place.

def transform_ast(ast, transform_node):
    if getattr(ast, "first", None) is not None:
        ast.first = transform_ast(ast.first, transform_node)
    if getattr(ast, "second", None) is not None:
        ast.second = transform_ast(ast.second, transform_node)
    if getattr(ast, "third", None) is not None:
        ast.third = transform_ast(ast.third, transform_node)
    return transform_node(ast)


# Mark the known symbols as '(vector)' or '(scalar)', and
# include the NumPy type information for that type.
def tag_symbols(node, symbol_table):
    def transform_symbol_to_vector(node):
        if node.id == "(symbol)":
            if node.value in symbol_table:
                datatype, typeinfo = symbol_table[node.value]
                node.id = datatype
                node.typeinfo = typeinfo
                if datatype == "(vector)":
                    # Initialize this so I can use it later on as a pivot
                    node.first = None
        return node

    return transform_ast(node, transform_symbol_to_vector)

# Only allow "&" for scalar references.
def clean_deref(ast):
    def transform_deref(node):
        if node.id == "&":
            assert node.first.id == "(scalar)", node.first.id
        return node

    return transform_ast(ast, transform_deref)

# If x is a vector, then "x+offset_x" will be converted into
#   x[offset_x:].ctypes.data_as(typeinfo)
# What I do is transform:
#   ('+'  ('(vector)' 'x') ('(scalar)' 'offset_x'))
# into
#   ('vector' 'x' ('(scalar)' 'offset_x'))
# This is a pivot of the AST.
def use_vector_addition(ast):
    return transform_ast(ast, _transform_vector_addition)

# Helper function
def _transform_vector_addition(node):
    # Binary "+'
    if node.id == "+" and node.second is not None:
        # with a vector on the LHS
        if node.first.id == "(vector)":
            vec = node.first
            if vec.first is None:
                vec.first = node.second
                return vec
            s = symbol("+")()
            s.first = vec.first
            s.second = node.second
            return s
        
    return node

# Special code for the "instrinsics"; len, abs, and shape

def fix_intrinsics(ast):
    return transform_ast(ast, _transform_intrinsic_node)

def _transform_intrinsic_node(node):
    if node.id == "(":
        assert node.first.id == "(symbol)", node.first.id
        fname = node.first.value
        if fname == "len":
            if node.second.id != "(vector)":
                raise AssertionError("len() only works on vectors")
            node.second.id = "(symbol)" # XXX? Do this to prevent rewrites on ouput
        elif fname == "abs":
            if node.second.id == "(vector)":
                raise AsertionError("abs() does not work on vectors")
        elif fname == "shape":
            # ??? What does this do?
            assert node.second.id == ",", "shape takes two parameters"
            assert node.second.second.id != ",", "shape takes only two parameters"
        else:
            #raise AssertionError("Unknown function %r" % (fname,))
            pass
    return node
    

# Convert a string containing a C expression into a Python expression.
# "symbol_table" contains scalar/vector information
# and numpy type information
def convert_c_expression(expr, symbol_table):
    ast = None
    try:
        #print "Process", repr(expr)
        ast = parse(expr)
        #print "AST", ast
        ast = tag_symbols(ast, symbol_table)
        ast = fix_intrinsics(ast)
        ast = clean_deref(ast)
        ast = use_vector_addition(ast)

        codegen = EmitPythonExpression()
        s = codegen.emit_to_string(ast)
        return s
    except Exception:
        print "Process", repr(expr)
        print "AST", ast
        raise
