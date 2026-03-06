cd /workspace/
mkdir dave
cd dave/
git clone https://github.com/AI-for-Toxicity/PaperDataExtraction
cd PaperDataExtraction/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_train.txt
