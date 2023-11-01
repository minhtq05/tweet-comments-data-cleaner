import pandas as pd

def export_data_to_excel(data):
    data = pd.DataFrame(
        data, columns=['sentence_id', 'sentence_label', 'sentence_text'])

    data_0, data_1 = data[data['sentence_label']
                          == 0], data[data['sentence_label'] == 1]

    data_0.to_csv('cleaned_0.csv', index=False)
    data_1.to_csv('cleaned_1.csv', index=False)
