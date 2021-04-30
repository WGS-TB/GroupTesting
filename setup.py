from setuptools import setup

from group_testing import __version__, _program

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="GroupTesting",
    version=__version__,
    author="Hooman Zabeti, Nick Dexter",
    author_email="hzabeti@sfu.ca, nicholas_dexter@sfu.ca",
    description="Group testing for SARS-CoV-2 in large populations.",
    url="https://github.com/WGS-TB/GroupTesting",
    license="GPL-3",
    packages=["group_testing"],
    entry_points="""
    [console_scripts]
    {program} = group_testing.group_testing:main
    """.format(program=_program),
    install_requires=requirements,
    include_package_data=True,
)