from os.path import join

# Generate new datasets for Inflection:
training_mode = 'LEMMA'  # choose either 'FORM' or 'LEMMA'.
data_dir = join('..', 'LemmaSplitData')
tsv_dir = join('..', 'LemmaSplitData', f'{training_mode}_TSV_FORMAT')

# Choose one of the following groups
languages1 = ['tgk', 'dje', 'mao', 'lin', 'xno', 'lud', 'zul', 'sot', 'vro', 'ceb', 'mlg', 'gmh', 'kon', 'gaa', 'izh',
              'mwf', 'zpv', 'kjh', 'hil', 'gml', 'tel', 'vot', 'czn', 'ood', 'mlt', 'gsw', 'orm', 'tgl', 'sna', 'frr',
              'syc', 'xty', 'ctp', 'dak', 'liv', 'aka', 'ben', 'nya', 'cly', 'swa', 'lug', 'bod', 'kan', 'kir', 'cre',
              'pus', 'lld', 'ast', 'crh', 'cpa', 'uig', 'fur', 'evn', 'aze', 'kaz', 'azg', 'urd', 'bak']
languages2 = ['pei', 'nno', 'vec', 'nob', 'dan', 'tuk', 'otm', 'ote', 'san', 'glg', 'frm', 'uzb', 'fas', 'est']
languages3 = ['ang', 'hin', 'nld', 'sme', 'olo', 'mdf', 'cat', 'isl', 'swe', 'kpv', 'mhr']
languages4 = ['myv', 'krl', 'eng', 'udm', 'vep', 'fin', 'deu']
all_languages = [languages1, languages2, languages3, languages4]

choice = 1
languages = all_languages[choice - 1]

log_file = join(f'log_file{choice}_{training_mode}.txt')

load_model = False
save_model = True

# Training hyperparameters
num_epochs = 50
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 256
num_layers = 1
encoder_dropout = 0.0
decoder_dropout = 0.0
comment = f"epochs={num_epochs} lr={learning_rate} batch={batch_size} embed={encoder_embedding_size} hidden_size={hidden_size}"

excel_results_file = f"ResultsFile{len(languages)}Langs{choice}.{training_mode}.xlsx"
