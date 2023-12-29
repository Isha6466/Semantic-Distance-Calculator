from setuptools import setup

setup(
    name='word2vec_similarity',
    version='0.1',
    packages=['word2vec_module'],
    install_requires=[
        'gensim',
        'scikit-learn',
        'numpy',
        # Add other dependencies
    ],
)
