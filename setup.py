import sys
import setuptools

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by ctc_score.')

setuptools.setup(
    name="ctc_score",
    version='0.1.0.a3',
    url="https://github.com/tanyuqian/ctc-gen-eval",
    author="Mingkai Deng*, Bowen Tan*, Zhengzhong Liu, Eric P. Xing, Zhiting Hu, Yuheng Zha",
    description="CTC: A Unified Framework for Evaluating Natural Language Generation",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='CTC Score',
    license='MIT',

    packages=setuptools.find_packages(),
    install_requires=[
        'nltk',
        'transformers>=4.3',
        'datasets>=1.5',
        'cleantext',
        'bert_score>=0.3',
    ],
    entry_points={
        'console_scripts': [
            "ctc_score=ctc_score_cli.score:main",
        ]
    },
    include_package_data=True,
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
