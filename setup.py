import setuptools

setuptools.setup(
    name="grc",
    packages=setuptools.find_packages(),

    # Dependencies from GRC CMakeLists.txt
    install_requires=[
        "pyYAML",
        "mako",
        "pygobject",
        "numpy",
        "jsonschema",
    ]
)