from setuptools import find_packages, setup

package_name = 'od_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/coco_label.json']),  
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey13',
    maintainer_email='rokey13@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = od_package.od_task:main',
        ],
    },
)
