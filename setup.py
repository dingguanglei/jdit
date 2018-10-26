from setuptools import setup

setup(
    name="jdit",  # pypi中的名称，pip或者easy_install安装时使用的名称，或生成egg文件的名称
    version="0.0.1",
    author="Guanglei Ding",
    author_email="dingguanglei.bupt@qq.com",
    description=("This is a framework for research based on pytorch"),
    license="Apache License 2.0",
    keywords="pytorch research framework",
    url="https://github.com/dingguanglei/jdit",
    packages=['jdit','jdit/trainer','jdit/trainer/gan','jdit/trainer/instances',
              'mypackage','mypackage/model','mypackage/metric','mypackage/model/shared'],  # 需要打包的目录列表

    # 需要安装的依赖
    install_requires=[
        # 'pip>=18.0'
        # 'torch>=0.4.1',
        # 'setuptools>=16.0',
        # 'torchvision',
        # 'psutil>=5.4.6',
        # 'tqdm>=4.23.4',
        # 'pandas>=0.23.0',
        # "tensorboardX>=1.4",
        # # 'tensorboard>=1.7.0',
        "nvidia_ml_py3>=7.352.0",
        # "pandas>=0.23.1",
        # 'numpy>=1.14.5',
        'imageio>=2.4.1'
    ],

    # # 添加这个选项，在windows下Python目录的scripts下生成exe文件
    # # 注意：模块与函数之间是冒号:
    # entry_points={'console_scripts': [
    #     'redis_run = RedisRun.redis_run:main',
    # ]},

    # long_description=read('README.md'),
    classifiers=[  # 程序的所属分类列表
        "Development Status :: 1 - Alpha",
        "Topic :: Utilities",
        "License :: MIT :: GNU General Public License (MIT)",
    ],
    # 此项需要，否则卸载时报windows error
    zip_safe=False
)
