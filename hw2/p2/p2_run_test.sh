### example for running the python files, you don't have to submit this file

### Manually put your best model under folder checkpoint/ and rename as 'resnet18_best.pth' or 'mynet_best.pth'

### Inference (TA will run this on public p2_data/val and private p2_data/test with both model types)
python3 p2_inference.py --test_datadir ../hw2_data/p2_data/val --model_type resnet18 --output_path ./output/pred.csv
# python3 p2_inference.py --test_datadir ../hw2_data/p2_data/val --model_type mynet --output_path ./output/pred.csv

### Evaluation (TA will also run this)
python3 p2_eval.py --csv_path ./output/pred.csv --annos_path ../hw2_data/p2_data/val/annotations.json
