import sys
import setuptools

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by ctc_score.')

setuptools.setup(
    name="ctc_score",
    url="https://github.com/tanyuqian/ctc-gen-eval",
    install_requires=[
        'distance',
        'nltk',
        'transformers==4.3.3',
        'pytorch_lightning==1.2.6',
        'fairseq==0.10.0',
        'spacy==2.3.0',
        'datasets==1.5.0',
        'summa',
        'cleantext',
        'benepar',
        'bert_score==0.3.9',
        'texar_pytorch==0.1.3'
    ],
    entry_points={
        'console_scripts': [
            "ctc_score=ctc_score_cli.score:main",
        ]
    },
)
