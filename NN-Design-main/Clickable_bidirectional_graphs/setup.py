from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="Clickable_bidirectional_graphs",
    version="0.1.0",
    author="Deborah Aina",
    author_email="deborah.aina@outlook.com",
    description="streamlit component that accepts click events and sends data back to the streamlit app to be used",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    classifiers=["License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"],
    python_requires=">=3.7",
    install_requires=["streamlit>=1.2", "jinja2"],
)
