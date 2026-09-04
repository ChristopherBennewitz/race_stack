from setuptools import setup
import os
from glob import glob

package_name = 'id_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'rviz'),
            glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jonathan Becker',
    maintainer_email='JonathanBecker.Tech@gmail.com',
    description='The controller for sysid experiments',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_take = id_controller.run_take:main',
            'record_teleop = id_controller.record_teleop:main',
            'play_take = id_controller.play_take:main',
            'preflight = id_controller.preflight:main',
            'takes = id_controller.takes:main',
        ],
    },
)
