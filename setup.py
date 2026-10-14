from glob import glob

from setuptools import find_packages, setup

package_name = 'lavacone_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hyeonjin Woo',
    maintainer_email='guswls264664@gmail.com',
    description='YOLO 라바콘 검출과 2D 라이다 정합 (카메라-라이다 융합, 콘 추적, 외부 파라미터 확인)',
    license='AGPL-3.0-only',
    entry_points={
        'console_scripts': [
            'scan_probe = lavacone_fusion.scan_probe:main',
            'lidar_projection = lavacone_fusion.lidar_projection:main',
            'cone_detector = lavacone_fusion.cone_detector:main',
            'cone_fusion = lavacone_fusion.cone_fusion:main',
            'extrinsic_check = lavacone_fusion.extrinsic_check:main',
        ],
    },
)
