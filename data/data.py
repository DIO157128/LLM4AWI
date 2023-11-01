import pandas as pd

def split(data_type):
    df = pd.read_csv('raw/awi_{}.csv'.format(data_type))
    rq1_source = df['warning_method'].tolist()
    rq1_target = df['final_label'].tolist()
    rq1_target = [1 if t =='TP' else 0 for t in rq1_target]
    rq1_df = pd.DataFrame()
    rq1_df['source'] = rq1_source
    rq1_df['target'] = rq1_target
    rq1_df.to_csv('awi_{}.csv'.format(data_type),index=False)

    rq2_source = df['warning_line'].tolist()
    rq2_target = df['final_label'].tolist()
    rq2_target = [1 if t == 'TP' else 0 for t in rq2_target]
    rq2_df = pd.DataFrame()
    rq2_df['source'] = rq2_source
    rq2_df['target'] = rq2_target
    rq2_df.to_csv('awi_context_{}.csv'.format(data_type),index=False)

    rq2_source = df['warning_abstract_method'].tolist()
    rq2_target = df['final_label'].tolist()
    rq2_target = [1 if t == 'TP' else 0 for t in rq2_target]
    rq2_df = pd.DataFrame()
    rq2_df['source'] = rq2_source
    rq2_df['target'] = rq2_target
    rq2_df.to_csv('awi_abstract_{}.csv'.format(data_type),index=False)

if __name__ == '__main__':
    split('train')
    split('val')
    split('test')