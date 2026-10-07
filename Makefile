# LayerWatch. Python 3.12 in .venv (uv), Node for the dashboard and the CDK CLI.
PY := .venv/bin/python
UV := $(shell command -v uv || echo ~/.local/bin/uv)
export JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION := 1

.PHONY: setup data train eval eval-real report pdf demo serve dashboard build-dashboard test synth deploy film all clean-data

setup:            ## python env + node deps
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -r requirements.txt
	$(UV) pip install --python $(PY) torch --index-url https://download.pytorch.org/whl/cpu
	cd dashboard && npm install
	cd infra && npm install

data:             ## synthetic dataset: 2,000 train + 400 test layers (+400 hold-out), about 3 to 5 minutes
	$(PY) -m layerwatch.synth.generate --out data/synth --train-builds 50 --test-builds 10 --holdout-builds 10

train:            ## U-Net on CPU, 4 threads, early stop, export to ONNX, check cv.dnn == torch
	$(PY) -m layerwatch.train.train --minutes 40 --iters-per-epoch 40 --patience 6

eval:             ## metrics on the hold-out split and the dev test split, then the real images
	$(PY) -m layerwatch.eval.run_eval holdout
	$(PY) -m layerwatch.eval.run_eval test
	$(MAKE) eval-real

eval-real:        ## real powder bed images (needs data/real, see README)
	$(PY) -m layerwatch.eval.real_eval

report:           ## docs/LayerWatch_Report.md + .html + figures, then the PDF
	$(PY) -m layerwatch.eval.report
	node docs/render_pdf.mjs

demo:             ## the 480 layer demo build through the handler, writes dashboard/public/run
	$(PY) -m layerwatch.demo.run_build

serve: build-dashboard  ## the Lambda handler behind FastAPI + the dashboard on http://localhost:8304
	$(PY) -m uvicorn server.app:app --port 8304

dashboard:        ## dashboard dev server on http://localhost:5304 (proxies /api to 8304)
	cd dashboard && npm run dev

build-dashboard:
	cd dashboard && npm run build

test:
	$(PY) -m pytest -q tests

synth:            ## CloudFormation for the AWS stack (no credentials needed)
	cd infra && npx cdk synth --quiet

deploy:           ## needs AWS credentials and Docker (arm64 image build)
	cd infra && npx cdk deploy --require-approval never $(if $(EMAIL),-c alarmEmail=$(EMAIL),)

film:             ## needs `make dashboard` running in another terminal
	node output/record.mjs

all: data train eval demo report test synth
