from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='xtorch-bridge',
    version='0.1.0',
    author='Your Name',
    author_email='your@email.com',
    description='Bridge between PyTorch and xtorch native C++ library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=['xtorch_bridge'],
    ext_modules=[
        CppExtension(
            name='xtorch_native',
            sources=['cpp/xtorch_wrapper.cpp'],
            extra_compile_args=['-O3'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=['torch'],
    python_requires='>=3.7',
)