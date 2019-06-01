from setuptools import setup

setup(
    name="jdit",  # pypi中的名称，pip或者easy_install安装时使用的名称，或生成egg文件的名称
    version="0.0.14",
    author="Guanglei Ding",
    author_email="dingguanglei.bupt@qq.com",
    maintainer='Guanglei Ding',
    maintainer_email='dingguanglei.bupt@qq.com',
    description=("Make it easy to do research on pytorch"),
    # long_description=open('docs/source/index.rst').read(),
    long_description=open('README.md').read(),
        long_description_content_type="text/markdown",
    license="Apache License 2.0",
    keywords="pytorch research framework",
    platforms=["all"],
    url="https://github.com/dingguanglei/jdit",
    packages=['jdit','jdit/trainer',
              'jdit/trainer/gan',
              'jdit/trainer/single',
              'jdit/trainer/instances',
              'jdit/assessment',
              'jdit/parallel'],
              # 'mypackage','mypackage/model','mypackage/metric','mypackage/model/shared'],  # 需要打包的目录列表

    # 需要安装的依赖
    install_requires=[
        "nvidia_ml_py3>=7.352.0",
        'imageio',
    ],

    # # 添加这个选项，在windows下Python目录的scripts下生成exe文件
    # # 注意：模块与函数之间是冒号:
    # entry_points={'console_scripts': [
    #     'redis_run = RedisRun.redis_run:main',
    # ]},

    # long_description=read('README.md'),
    classifiers=[  # 程序的所属分类列表
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        # 'Development Status :: 2 - Pre-Alpha',
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        "Programming Language :: Python :: 3 :: Only",
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization  '
    ],
    # 此项需要，否则卸载时报windows error
    zip_safe=False
)
