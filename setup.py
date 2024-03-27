from setuptools import setup


def read_requirements():
    with open('requirements.txt', 'r') as req:
        content = req.read()
        requirements = content.split('\n')

    return requirements


setup(
    name='npp',
    version='1.0',
    packages=['tm_module'],
    url='',
    license='',
    author='Antônio Pereira',
    author_email='antonio258p@gmail.com',
    description='Python topic modeling tool',
    install_requires=read_requirements()
)