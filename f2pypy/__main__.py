import argparse

from . import pypy_shim
from . import codegen


from .f2py.crackfortran import crackfortran

# emulate the f2py command-line parameters?

parser = argparse.ArgumentParser(
    description="Generate Python/ctypes interface file to a shared library")

parser.add_argument("definition_file", 
                    help="the pyf definition filename")
parser.add_argument("-l", "--library", action="append",
                    help="shared library to search")

parser.add_argument("--skip", action="append", default=[],
                    help="skip these functions")

parser.add_argument("--only", action="append", default=[],
                    help="only convert these functions")

# Do I really need this? Or do I need something more complicated?
parser.add_argument("--rename-template", action="store", 
                    default="{0}",
                    help="rename pattern")

def main():
    args = parser.parse_args()
    if not args.library:
        parser.error("Must specify one or more shared library files")

    if args.skip and args.only:
        parser.error("Cannot specify both --skip and --only")
    skip = set()
    for name in args.skip:
        skip.update(name.split(","))
    only = set()
    for name in args.only:
        only.update(name.split(","))
        
    ast = crackfortran(args.definition_file)
    for module in ast:
        assert module["block"] == "python module"
        output_filename = module["name"] + ".py"
        with open(output_filename, "w") as outfile:
            print "Generating code to", repr(output_filename)
            gen = codegen.CodeGen(outfile, libraries=args.library,
                                  rename_template=args.rename_template,
                                  skip=skip, only=only)
            gen.emit_python_module(module)

if __name__ == "__main__":
    main()
