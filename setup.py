from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='protopipe',
      version='0.1',
      description='Pipeline to process events from DL0 to DL3',
      url='http://github.com/jjlk/protopipe',
      author='CEA',
      author_email='julien.lefaucheur@cea.fr',
      license='MIT',
      packages=find_packages(),
      install_requires=['ctapipe'],
      zip_safe=False)