import os
import sys
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import numpy as _np
import pybind11 as _pybind11

__version__ = "0.1.0"


def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    if has_flag(compiler, '-std=c++17'):
        return '-std=c++17'
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')


class BuildExt(build_ext):
    compiler_flag_native = '-march=native'
    c_opts = {
        'msvc': ['/EHsc', '/openmp', '/O2'],
        'unix': ['-O3', compiler_flag_native],
    }
    link_opts = {
        'unix': [],
        'msvc': [],
    }

    if os.environ.get('HNSWLIB_NO_NATIVE'):
        c_opts['unix'].remove(compiler_flag_native)

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        link_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    else:
        c_opts['unix'].append('-fopenmp')
        link_opts['unix'].extend(['-fopenmp', '-pthread'])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = BuildExt.c_opts.get(ct, [])[:]
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
            if not os.environ.get('HNSWLIB_NO_NATIVE'):
                if not has_flag(self.compiler, BuildExt.compiler_flag_native):
                    try:
                        opts.remove(BuildExt.compiler_flag_native)
                    except ValueError:
                        pass
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\"%s\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args = ext.extra_compile_args or []
            ext.extra_link_args = ext.extra_link_args or []
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(BuildExt.link_opts.get(ct, []))

        build_ext.build_extensions(self)


include_dirs = [
    _pybind11.get_include(),
    _np.get_include(),
    os.path.join(os.path.dirname(__file__), 'include'),
    os.path.join(os.path.dirname(__file__), 'python_bindings'),
]

source_files = [
    os.path.join('python_bindings', 'rabitq_bindings.cpp'),
    os.path.join('python_bindings', 'hnsw_bindings.cpp'),
    os.path.join('python_bindings', 'ivf_bindings.cpp'),
    os.path.join('python_bindings', 'symqg_bindings.cpp'),
]

ext_modules = [
    Extension(
        'rabitqlib._rabitqlib',
        sources=source_files,
        include_dirs=include_dirs,
        language='c++',
    )
]

setup(
    name='rabitqlib',
    version=__version__,
    description='RaBitQ Python bindings',
    long_description='RaBitQ Python bindings for HNSW, IVF and SymQG',
    packages=['rabitqlib'],
    package_dir={'rabitqlib': 'python_bindings'},
    ext_modules=ext_modules,
    install_requires=['numpy'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
