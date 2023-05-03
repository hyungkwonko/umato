import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name="umato",
	version="0.1.2",
	author="Hyeon Jeon",
	author_email="hj@hcil.snu.ac.kr",
	description="Python Implementation of UMATO (Uniform Manifold Approximation with Two-Phase Optimization)", 
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/hyungkwonko/umato",
	classifiers=[
			"Programming Language :: Python :: 3",
			"License :: OSI Approved :: MIT License",
			"Operating System :: OS Independent",
	],
	install_requires=[
		"numpy",
		"numba",
		"scipy",
		"scikit-learn",
		"joblib",
		"matplotlib",
		"pandas",
		"fcsparser",
	],
	package_dir={"": "src"},
	packages=setuptools.find_packages(where="src"),
	python_requires=">=3.9.0",
)