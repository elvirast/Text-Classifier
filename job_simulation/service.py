import pickle
import flask
import argparse
import marshmallow
import marshmallow.validate
#from fasttext import load_model
import cnf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from inference import inference_svm, inference_fasttext
app = flask.Flask(__name__)

SUPPORTED_MODELS = ('tfidf', 'fasttext', 'bert', 'shallow')


class ClassificationPayloadSchema(marshmallow.Schema):
	text = marshmallow.fields.String(required=True)
	model = marshmallow.fields.String(required=True, validate=marshmallow.validate.OneOf(SUPPORTED_MODELS))

clf_ft = None
clf_svm = None
vectorizer = None

classification_payload_schema = ClassificationPayloadSchema()


@app.route('/')
def index():
	return flask.render_template('index.html')


@app.route('/resolve', methods=["POST"])
def resolve():

	try:
		payload = classification_payload_schema.load(flask.request.json)
	except marshmallow.ValidationError as e:
		out = {'error': f'missing expected params: {e.messages}'}
		return flask.jsonify(out)
	text = payload['text']
	model = payload['model']
	try:

		if model== 'tfidf':
			classification = inference_svm(text, vectorizer, clf_svm)  # type: dict
			#classification = {cnf.map_dct[str(key)]: '' }
		#elif model == 'fasttext':
		#	classification = inference_fasttext(text, clf_ft)
		else:
			classification = {str(i): 0.1 for i in range(7)}
		out = {'text': text, 'model': model, 'classification': classification}
	except Exception as e:
		out = {'error': str(e)}
	return flask.jsonify(out)


def load_fasttext_p(path = cnf.fasttext_paths['bin']):
	global clf_ft
	clf_ft = load_model(path)


def load_svm_p(path = cnf.svm_paths['clf']):
	global clf_svm
	with open(path, 'rb') as f:
		clf_svm = pickle.load(f)
	f.close()


def load_tfidf_p(path = cnf.svm_paths['vectorizer']):
	global vectorizer
	with open(path, 'rb') as f:
		vectorizer = pickle.load(f)
	f.close()

if __name__ == '__main__':
	#load_fasttext_p()
	load_tfidf_p()
	load_svm_p()
	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', dest='debug', action='store_true')
	parser.add_argument('-p', '--port', dest='port', type=int, default=8010)
	args = parser.parse_args()

	app.run(host='0.0.0.0', port=args.port, debug=args.debug)
