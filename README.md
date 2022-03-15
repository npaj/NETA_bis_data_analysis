# NETA_bis_data_analysis
Analyse data from neta bis acquisition soft


Requirements
--
- argparse
- scipy
- tqdm
- pandas
- numpy
- matplotlib
- plotly
- joblib
- multiprocessing
- dash
- dash_core_components
- dash_html_components


Usage
--
## processing data : 

    python convert_data/convert_txt2npy.py --path path/to/data
    python convert_data/metadata_txt2dataframe.py --path path/to/data

## application : 

    python dash-app.py --path path to/processed/data