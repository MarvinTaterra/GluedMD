from setuptools import setup, Extension
import os, platform

openmm_dir = os.environ.get('OPENMM_DIR', '/usr/local/openmm')

setup(
    name='gluedplugin',
    version='@GLUED_VERSION@',
    py_modules=['gluedplugin'],
    ext_modules=[
        Extension(
            name='_gluedplugin',
            sources=['gluedpluginwrap.cpp'],
            include_dirs=[
                os.path.join(openmm_dir, 'include'),
            ],
            library_dirs=[
                os.path.join(openmm_dir, 'lib'),
                os.path.join(openmm_dir, 'lib', 'plugins'),
            ],
            libraries=['OpenMM', 'OpenMMGlued'],
        )
    ]
)
