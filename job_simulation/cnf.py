map_dct = { '0': 'bank_service',
            '1': 'credit_card',
            '2':  'credit_reporting',
            '3': 'debt_collection',
            '4': 'loan',
            '5': 'money_transfers',
            '6': 'mortgage'}
fasttext_paths = {
    'bin': './job_simulation/models/fasttext/ft_best.bin',
    'vec': './job_simulation/models/fasttext/ft_best.vec',
    }
svm_paths = {'clf': './job_simulation/models/svm/sgd_best.pickle',
            'vectorizer': './job_simulation/models/svm/tfidf_word_1000.pickle'}