

def preprocess_data(df, config):
    train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
        {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
    train.columns = ['spec_id','min']

    tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
        {'spectrogram_label_offset_seconds':'max'})
    train['max'] = tmp

    tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
    train['patient_id'] = tmp

    tmp = df.groupby('eeg_id')[config.class_columns].agg('sum')
    for t in config.class_columns:
        train[t] = tmp[t].values
        
    y_data = train[config.class_columns].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[config.class_columns] = y_data

    tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
    train['target'] = tmp

    train = train.reset_index()
    print('Train non-overlapp eeg_id shape:', train.shape )
    print(train.head())
    return train