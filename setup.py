import sys
import setuptools

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by ctc_score.')

setuptools.setup(
    name="ctc_score",
    version='0.1.0',
    url="https://github.com/tanyuqian/ctc-gen-eval",
    install_requires=[
        'nltk',
        'transformers==4.3.3',
        'datasets==1.5.0',
        'cleantext',
        'bert_score==0.3.9',
    ],
    entry_points={
        'console_scripts': [
            "ctc_score=ctc_score_cli.score:main",
        ]
    },
)
