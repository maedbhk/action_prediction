from setuptools import find_packages, setup

setup(
    name='action_prediction',
    packages=find_packages(),
    version='0.1.0',
    description='Investigating social and action prediction using eye tracking and behavioral performance',
    entry_points={
    'console_scripts': [
        'transfer-data-from-savio=analysis_code.data_transfer:data_from_savio',
        ]
    },
    author='Shannon Lee and Maedbh King',
    license='MIT',
)
