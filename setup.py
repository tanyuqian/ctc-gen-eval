import sys
import setuptools

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by CTC.')

setuptools.setup(
    name="ctc_score",
    version="0.0.1",
    url="https://github.com/tanyuqian/ctc-gen-eval",
    description="Evaluation"
)
