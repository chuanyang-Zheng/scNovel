import setuptools

setuptools.setup(
    name = "scNovel",
    version = "1.2.0",
    description = "scNovel, a neural network framework, "
                  "provides a fast and reliable tools for scRNA-seq novel rare cell detection.",
    license = "MIT Licence",
    python_requires=">=3.5.0",
    packages=setuptools.find_packages(),
    url = "https://github.com/chuanyang-Zheng/scNovel",
    author = "Chuanyang Zheng",
    author_email = "cyzheng21@cse.cuhk.edu.hk"
)