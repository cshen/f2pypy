from string import Template

from . import parse_c_expr

typespec_as_ctypes_ptr = {
    "real": "_ct.POINTER(_ct.c_float)",
    "double precision": "_ct.POINTER(_ct.c_double)",
    "integer": "_ct.POINTER(_ct.c_int)",
    "complex": "_ct.POINTER(_complex_float)",
    "double complex": "_ct.POINTER(_complex_double)",
    }
proto_as_ctypes_ptr = {
    "float*": "_ct.POINTER(_ct.c_float)",
    "double*": "_ct.POINTER(_ct.c_double)",
    "int*": "_ct.POINTER(_ct.c_int)",
    "complex_float*": "_ct.POINTER(_complex_float)",
    "complex_double*": "_ct.POINTER(_complex_double)",
    "char*": "_ct.c_char_p",
    }

typespec_as_ctypes_value = {
    "real": "_ct.c_float",
    "double precision": "_ct.c_double",
    "integer": "_ct.c_int",
    "complex": "_complex_float",
    "double complex": "_complex_double",
    }

_typespec_python_to_ctypes_var_typemap = {
    "real": "_ct.c_float",
    "double precision": "_ct.c_double",
    "integer": "_ct.c_int",
    "complex": "_to_complex_float",
    "double complex": "_to_complex_double",
}
_typespec_to_numpy_letter = {
    "real": "f",
    "double precision": "d",
    "integer": "i",
    "complex": "F",
    "double complex": "D",
    }
def convert_from_input(varname, var, expr=None, is_input=None):
    if is_input is None:
        is_input = "in" in var["intent"]
    typespec = var["typespec"]
    if expr is None:
        expr = varname
    
    if "dimension" not in var:
        # This is scalar data. Use a ctypes value
        ct_datatype = _typespec_python_to_ctypes_var_typemap[typespec]
        if is_input:
            return "%s = %s(%s)" % (varname, ct_datatype, expr)
        else:
            return "%s = %s()" % (varname, ct_datatype)

    numpy_letter = _typespec_to_numpy_letter[typespec]
    if is_input:
        args = [expr, repr(numpy_letter)]
        if "copy" in var["intent"]:
            args.append("copy=overwrite_%s" % varname)
        return "%s = _np.array(%s)" % (varname, ", ".join(args))
    else:
        dimension = var["dimension"]
        assert dimension != ["*"], var
        dims = ", ".join(dimension)
#        if len(dimension) == 1:
#            dims += "," # force this to be a tuple
        return "%s = _np.zeros((%s), %r)" % (varname, dims, numpy_letter)

def cast_to_fortran(varname, var):
    if "dimension" not in var:
        ndims = 0
    else:
        ndims = len(var["dimension"])

    if ndims == 0:
        return varname
    else:
        typespec = var["typespec"]
        return "%s.ctypes.data_as(%s)" % (varname, typespec_as_ctypes_ptr[typespec])


typespec_ctypes_var_to_python = {
    "real": "{0}.value",
    "double precision": "{0}.value",
    "integer": "{0}.value",
    "complex": "complex({0}.real,  {0}.imag)",
    "double complex": "complex({0}.real,  {0}.imag)",
}

def convert_to_output(varname, var):
    if "dimension" not in var:
        ndims = 0
    else:
        ndims = len(var["dimension"])
    typespec = var["typespec"]

    assert "out" in var["intent"]
    if ndims == 0:
        return typespec_ctypes_var_to_python[typespec].format(varname)
    else:
        return varname
    


#### A quick template language

class TemplateWrapper(object):
    def __init__(self, value, filters):
        self.value = value
        self.filters = filters
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return repr(self.value)
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        f = self.filters[name]
        return TemplateWrapper(f(self.value), self.filters)
    
    def __getitem__(self, name):
        return TemplateWrapper(self.value[name], self.filters)


#####

def _find_args(func, intent_filter=None):
    vars = func["vars"]
    for varname in func["args"]:
        var = vars[varname]
        if intent_filter is None:
            yield (varname, var)
        else:
            if intent_filter in var["intent"]:
                yield (varname, var)


######## Some helper functions for string template evaluation

# Given f2py function information, make the ctypes ".argtypes" attribute for it
def argtypes(func):
    # First, get it from the callprotoargument, if it exists
    # 
    # 'f2pyenhancements': {'callprotoargument': 'int*,float*,int*,float*,int*,float*,float*',
    try:
        proto = func["f2pyenhancements"]["callprotoargument"]
    except KeyError:
        pass
    else:
        fields = proto.split(",")
        # For functions, f2py puts in the return value as the first paramter.
        # Using ctypes (at least for my system), I don't need to do that.
        if func["block"] == "function" and "_return_value" not in func["f2pyenhancements"]["callstatement"]:
            del fields[0]
        terms = [proto_as_ctypes_ptr[field] for field in fields]
        return "[" + ", ".join(terms) + "]"

    # If it doesn't exist, then assume it's the same as the Fortran parameters.
    terms = []
    for arg in func["args"]:
        var = func["vars"][arg]
        terms.append(typespec_as_ctypes_ptr[var["typespec"]])
    return "[" + ", ".join(terms) + "]"

# Given f2py function information, make the Python call parameters.
# These can be in a different order because Python supports default values.
# These can include new parameters, because each "copy" option
# adds an "overwrite_" parameter.
def argspec(func):
    required_terms = []
    optional_terms = []
    extra_terms = []
    for (varname, var) in _find_args(func, "in"):
        if "optional" in var["attrspec"]:
            optional_terms.append( "%s = None" % (varname,) )
        else:
            required_terms.append( varname )
        if "copy" in var["intent"]:
            extra_terms.append( "overwrite_%s = 0" % (varname,))
    
    return ", ".join(required_terms + optional_terms + extra_terms)


# Given f2py function information, figure out the named used in the
# shared library.
# XXX This is incomplete! The actual name depends on the compiler.
# Is it possible to autodetect this by looking at the shared library symbols?
def raw_fortranname(func):
    enhancements = func.get("f2pyenhancements", {})
    if "fortranname" in enhancements:
        s = enhancements["fortranname"]
        # XXX In f2py this depends on F_FUNC, which is user-defined
        # For now I'll just strip out the call to f_func()
        if s.startswith("f_func(") and s.endswith(")"):
            s = s[7:-1]
            return s.split(",")[0]
    return func["name"]

def _get_symbol_table(func):
    # TODO: also add the return function name
    symbol_table = {}
    for varname, var in func["vars"].items():
        if "dimension" in var:
            symbol_table[varname] = ("(vector)", typespec_as_ctypes_ptr[var["typespec"]])
        else:
            symbol_table[varname] = ("(scalar)", typespec_as_ctypes_ptr[var["typespec"]])
    return symbol_table

            
def callargs(func):
    # 'f2pyenhancements': {'callstatement': '(*f2py_func)(&n,x+offx,&incx,y+offy,&incy,&c,&s)'},
    api_name = "_api_" + func["name"]
    try:
        stmt = func["f2pyenhancements"]["callstatement"]
    except KeyError:
        return "%s(%s)" % (api_name,
                           ", ".join(cast_to_fortran(varname, func["vars"][varname]) for varname in func["args"]))
    else:
        stmt = stmt.strip()
        if stmt[:1] == "{":
            assert stmt[-1:] == "}", stmt
            stmt = stmt[1:-1].strip()
        if stmt[-1:] == ";":
            stmt = stmt[:-1]
            stmt = stmt.strip()
        
        assert "(*f2py_func)" in stmt, stmt
        stmt = stmt.replace("(*f2py_func)", api_name)
        py_stmt = parse_c_expr.convert_c_expression(stmt, _get_symbol_table(func))
        return py_stmt
        

def returnargs(func):
    terms = []
    for (funcname, func) in _find_args(func, "out"):
        fmt = convert_to_output(funcname, func)
        terms.append(fmt.format(funcname))
    return ", ".join(terms)


class CodeGen(object):
    def __init__(self, outfile, libraries, rename_template, skip, only):
        self.outfile = outfile
        self._indentation = 0
        self._indent = ""
        self.filters = dict(repr = repr,
                            fortranname = self._fortranname,
                            argtypes = argtypes,
                            argspec = argspec,
                            callargs = callargs,
                            returnargs = returnargs)
        self.config = {"libraries": libraries,
                       "rename_template": rename_template,
                       "skip": skip,
                       "only": only}


    def _fortranname(self, func):
        name = raw_fortranname(func)
        return self.config["rename_template"].format(name)



    def indent(self):
        self._indentation += 1
        self._indent = "  "*self._indentation

    def dedent(self):
        assert self._indentation > 0
        self._indentation -= 1
        self._indent = "  "*self._indentation

    def raw_write(self, text):
        if "\n" not in text:
            self.outfile.write(self._indent + text + "\n")
            return
        lines = []
        for line in text.splitlines(True):
            lines.append(self._indent + line)
        assert lines, "Why did you add nothing?"
        self.outfile.writelines(lines)
        if not lines[-1].endswith("\n"):
            self.outfile.write("\n")

    def write(self, text, **kwargs):
        # Handle template conversion
        d = dict( (k, TemplateWrapper(v, self.filters)) for (k, v) in kwargs.items() )
        text = text.format(**d)
        self.raw_write(text)

    ###
    def emit_start(self, module):
        self.write("""\
# Wrapper {module[name]}.py automatically generated from f2pypy.

import ctypes as _ct
from ctypes.util import find_library as _find_library
import numpy as _np

# I don't think this is the right code
def _shape(arr, dim):
    return arr.shape[dim]

_search_libraries = {config[libraries].repr}

def _find_available_libraries():
    libraries = {{}}
    _unavilable = []
    for library_name in _search_libraries:
        if "/" in library_name or "\\\\" in library_name:
            path = library_name
        else:
            path = _find_library(library_name)

        module = None
        if path is not None:
            try:
                module = _ct.cdll.LoadLibrary(path)
            except OSError:
                pass
            
        if module is None:
            _unavilable.append(library_name)
        libraries[library_name] = module
        
    return libraries, _unavilable

_resolved_libraries, _unavilable_libraries = _find_available_libraries()

def _find_function(name):
    tried = []
    for library_name in _search_libraries:
        library = _resolved_libraries[library_name]
        if library is not None:
            try:
                return getattr(library, name)
            except AttributeError:
                tried.append(library_name)
    if _unavilable_libraries:
        raise ImportError("Could not find %r in %s; could not load %s" %
                          (name, ", ".join(map(repr, tried)), ", ".join(map(repr, _unavilable_libraries))))
    else:
        raise ImportError("Could not find %r in %s" %
                          (name, ", ".join(map(repr, tried))))

class _complex_float(_ct.Structure):
    _fields_ = [("real", _ct.c_float), ("imag", _ct.c_float)]
class _complex_double(_ct.Structure):
    _fields_ = [("real", _ct.c_double), ("imag", _ct.c_double)]
def _to_complex_float(x=None):
    if x is None:
        return _complex_float()
    return _complex_float(x.real, x.imag)
def _to_complex_double(x=None):
    if x is None:
        return _complex_double()
    return _complex_double(x.real, x.imag)
    
""", module=module, config=self.config)

    def emit_end(self):
        self.write("# Bye for now.")

    def emit_python_module(self, module):
        self.emit_start(module)
        assert len(module["body"]) == 1
        for term in module["body"][0]["body"]:
            name = term["name"]
            if name in self.config["skip"]:
                continue
            if self.config["only"] and name not in self.config["only"]:
                continue
            
            if term["block"] == "function":
                self.emit_function(term)
            elif term["block"] == "subroutine":
                self.emit_subroutine(term)
            else:
                raise AssertionError((term["block"], term))
            self.write("\n")

    def emit_function(self, func):
        self.write("_api_{func[name]} = _f = _find_function({func.fortranname.repr})", func=func)
        self.write("_f.argtypes = {func.argtypes}", func=func)
        name = func["result"]
        typespec = func["vars"][name]["typespec"]
        self.raw_write("_f.restype = {0}".format(typespec_as_ctypes_value[typespec]))
        self.write("")

        # I don't know what f2py does if there's a return value plus items
        # with "output" intent.
        for var in func["vars"].values():
            assert "out" not in var.get("intent", []), ("Don't know how to handle", func)

        self.emit_def(func)
        self.indent()
        self.emit_initargs(func)
        self.emit_function_call(func)
        self.dedent()
        

    def emit_subroutine(self, func):
        self.write("_api_{func[name]} = _f = _find_function({func.fortranname.repr})", func=func)
        self.write("_f.argtypes = {func.argtypes}", func=func)
        self.write("_f.restype = None")
        assert "result" not in func
        self.emit_def(func)
        self.indent()
        self.emit_initargs(func)
        self.emit_subroutine_call(func)
        self.emit_subroutine_return(func)
        self.dedent()

    def emit_def(self, func):
        self.write("def {func[name]}({func.argspec}):", func=func)

    def emit_subroutine_call(self, func):
        self.write(callargs(func))

    def emit_subroutine_return(self, func):
        self.write("return {func.returnargs}", func=func)

    def emit_function_call(self, func):
        orig = stmt = func["f2pyenhancements"]["callstatement"]
        stmt = stmt.strip()
        name = func["name"]
        return_value_name = "%s_return_value" % (name,)
        if stmt.startswith(return_value_name):
            stmt = stmt[len(return_value_name):].lstrip()
            assert stmt[:1] == '=', orig
            assert stmt[:2] != '==', orig
            stmt = stmt[1:].lstrip()

            func["f2pyenhancements"]["callstatement"] = stmt
        else:
            result_ref = "(&"+name+","
            assert result_ref in stmt, (result_ref, orig)
            stmt = stmt.replace(result_ref, '(')
            func["f2pyenhancements"]["callstatement"] = stmt
        
        py_stmt = callargs(func)
        self.write("return " + py_stmt)
            
    def emit_initargs(self, func):
        symbol_table = _get_symbol_table(func)

        # "sortvars" contains the order in which parameters are
        # evaluated. For example, 'n' may be an optional value which
        # defaults to the length of 'x', so x need to be available (and
        # validated) first.

        for varname in func["sortvars"]:
            var = func["vars"][varname]
            if "hide" in var.get("intent", []):
                # This is a "hidden" variable, meaning that it's not part
                # public API. It's often used to compute values which are
                # used for bound checking.
                assert "in" not in var["intent"]
                c_expr = var.get("=", "Py_None") # XXX Huh? (copy&paste from below)
                py_expr = parse_c_expr.convert_c_expression(c_expr, symbol_table)
                self.write( convert_from_input(varname, var, py_expr, True))
                continue

            if "in" not in var.get("intent", []):
                # This function is only concerned with variable
                # initialization. I only need to track "hide" and "in"
                # variables.
                continue

            # Input variables
            if "optional" in var["attrspec"]:
                # Check if a parameter value was passed in. If not, use the default.
                self.write("if %s is None:" % (varname,))
                # XXX Huh? For some reason sgemv doesn't define an '='. What should
                # the default be for that case?
                c_expr = var.get("=", "Py_None")

                # The initializtion code is a C expression. Convert to Python.
                py_expr = parse_c_expr.convert_c_expression(c_expr, symbol_table)
                self.write("  " + convert_from_input(varname, var, py_expr))
                self.write("else:")
                self.write("  " + convert_from_input(varname, var))
            else:
                self.write(convert_from_input(varname, var))

            # Now that the variable has been defined, see if there is
            # a check() parameter. That also has embedded C code.
            if "check" in var:
                for c_expr in var["check"]:
                    py_expr = parse_c_expr.convert_c_expression(c_expr, symbol_table)
                    self.write("if not (%s):" % (py_expr,))
                    errmsg = "(%s) failed for argument %s: %s=%%s" % (c_expr, varname, varname)
                    self.write("  raise ValueError(%r %% %s.value)" % (errmsg, varname))

        for (funcname, func) in _find_args(func, "out"):
            if "in" in func["intent"]:
                continue
            self.write(convert_from_input(funcname, func))
