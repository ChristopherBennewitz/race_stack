import os
from glob import glob

from setuptools import setup

package_name = 'rl_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chris',
    maintainer_email='christopher.bennewitz@gmx.de',
    description='Deploys a PPO/RMA policy trained in f1tenth_parallel_sim as a drive controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_controller = rl_controller.rl_controller_node:main',
        ],
    },
)
