"""Command-line entry point: reproduce phases or score a CSV using the development-selected calibrated model."""
import argparse
import os
from pathlib import Path

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'2')
os.environ.setdefault('MPLBACKEND','Agg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['all','data','eda','unsupervised','models','evaluation','predict'])
    parser.add_argument('--force',action='store_true',help='Recompute requested phase(s); prerequisite phases reuse matching artifacts.')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,default=Path('artifacts/predictions/example_predictions.csv'))
    args = parser.parse_args()
    from readmission.pipeline import run_all,run_phase
    if args.phase=='all':
        run_all(args.force)
    elif args.phase=='predict':
        import joblib
        import pandas as pd
        from readmission.common import path,read_json,config
        if args.input is None:
            parser.error('predict requires --input path/to/encounters.csv')
        chosen = read_json('artifacts/model_choice.json')['experiment']
        model = joblib.load(path(f'artifacts/models/{chosen}__calibrated.joblib'))
        frame = pd.read_csv(args.input,keep_default_na=False,na_values=['?'])
        probability = model.predict_proba(frame)[:,1]
        result = pd.DataFrame({'input_row':range(len(frame)),'recorded_early_readmission_probability':probability,
                              'predicted_class':(probability>=config()['threshold']).astype(int),
                              'decision_threshold':config()['threshold']})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(args.output,index=False)
        print(result.to_string(index=False))
        print('Research demonstration on historical data; not a clinical decision tool.')
    else:
        run_phase(args.phase,args.force)


if __name__=='__main__':
    main()
