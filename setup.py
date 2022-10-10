import setuptools

setuptools.setup(
    name = "scRare",
    version = "1.1.0",
    description = "scRare, a neural network framework, "
                  "provides a fast and reliable tools for scRNA-seq novel rare cell detection.",
    license = "MIT Licence",
    python_requires=">=3.5.0",
    packages=setuptools.find_packages(),
    url = "https://github.com/chuanyang-Zheng/scRare",
    author = "Chuanyang Zheng",
    author_email = "cyzheng21@cse.cuhk.edu.hk"
)