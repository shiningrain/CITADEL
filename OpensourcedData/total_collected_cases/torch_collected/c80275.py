
import torch

g = torch.randn(10, dtype=torch.complex64, device='cuda')
g.requires_grad = True
opt = torch.optim.SGD([g], lr=1)

h = torch.ones(g.shape, device=g.device, dtype=g.dtype)
h[:1] = torch.abs(g[:1]).mean().to(dtype=g.dtype)

loss = h.mean()
loss.backward()


D:\Anaconda\envs\pyt\lib\site-packages\torch\autograd\__init__.py:173: UserWarning: Casting complex values to real discards
 the imaginary part (Triggered internally at  C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\Copy.cpp:239.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass

but `device='cpu'` gives


  File "D:\Anaconda\envs\pyt\lib\site-packages\torch\_tensor.py", line 363, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)

  File "D:\Anaconda\envs\pyt\lib\site-packages\torch\autograd\__init__.py", line 173, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass

RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn


Other combinations:

 - `device='cuda'` && `torch.abs(g)` -> `g`: no warning, no error
 - `device='cpu'` && `torch.abs(g)` -> `g`: same error
 - `[:1]` -> `[0]`: same warning
 - `[:1]` -> `[0]` && `torch.abs(g)` -> `g`: no warning, no error

"Complex to real" never happens in forward pass, and warning and error behaviors don't agree between devices. In full context, optimization proceeds fine despite the warning with `'cuda'`.

### Versions

PyTorch 1.11.0 via `conda-forge`, Python 3.8.12, Windows 10, i7 CPU


<details><summary><b>conda list</b></summary>


# packages in environment at D:\Anaconda\envs\pyt:
#
# Name                    Version                   Build  Channel
absl-py                   0.15.0             pyhd8ed1ab_0    conda-forge
aiohttp                   3.7.4.post0      py38h294d835_1    conda-forge
alabaster                 0.7.12                     py_0    conda-forge
anyio                     3.3.3            py38haa244fe_0    conda-forge
appdirs                   1.4.4              pyh9f0ad1d_0    conda-forge
argh                      0.26.2          pyh9f0ad1d_1002    conda-forge
argon2-cffi               21.1.0           py38h294d835_0    conda-forge
arrow                     1.2.0              pyhd8ed1ab_0    conda-forge
astroid                   2.5.8            py38haa244fe_0    conda-forge
async-timeout             3.0.1                   py_1000    conda-forge
async_generator           1.10                       py_0    conda-forge
atomicwrites              1.4.0              pyh9f0ad1d_0    conda-forge
attrs                     21.2.0             pyhd8ed1ab_0    conda-forge
audio2numpy               0.1.2                    pypi_0    pypi
audioread                 2.1.9            py38haa244fe_0    conda-forge
autopep8                  1.6.0              pyhd8ed1ab_1    conda-forge
babel                     2.9.1              pyh44b312d_0    conda-forge
backcall                  0.2.0              pyh9f0ad1d_0    conda-forge
backports                 1.0                        py_2    conda-forge
backports.functools_lru_cache 1.6.4              pyhd8ed1ab_0    conda-forge
bcrypt                    3.2.0            py38h294d835_1    conda-forge
binaryornot               0.4.4                      py_1    conda-forge
black                     22.3.0             pyhd8ed1ab_0    conda-forge
blas                      1.0                         mkl
bleach                    4.1.0              pyhd8ed1ab_0    conda-forge
blinker                   1.4                        py_1    conda-forge
boost                     1.77.0           py38he5193b3_0    conda-forge
boost-cpp                 1.77.0               h5b4e17d_1    conda-forge
brotli-python             1.0.9            py38h885f38d_5    conda-forge
brotlipy                  0.7.0           py38h294d835_1001    conda-forge
bzip2                     1.0.8                h8ffe710_4    conda-forge
ca-certificates           2022.4.26            haa95532_0
cached-property           1.5.2                hd8ed1ab_1    conda-forge
cached_property           1.5.2              pyha770c72_1    conda-forge
cachetools                4.2.4              pyhd8ed1ab_0    conda-forge
cairo                     1.16.0            hb19e0ff_1008    conda-forge
certifi                   2022.5.18.1      py38haa95532_0
cffi                      1.14.6           py38hd8c33c5_1    conda-forge
chardet                   4.0.0            py38haa244fe_1    conda-forge
charset-normalizer        2.0.0              pyhd8ed1ab_0    conda-forge
click                     8.1.3            py38haa244fe_0    conda-forge
cloudpickle               2.0.0              pyhd8ed1ab_0    conda-forge
cmarkgfm                  0.7.0            py38h294d835_0    conda-forge
colorama                  0.4.4              pyh9f0ad1d_0    conda-forge
colorednoise              2.0.0                    pypi_0    pypi
conda                     4.11.0           py38haa244fe_0    conda-forge
conda-package-handling    1.7.3            py38h31c79cd_1    conda-forge
configparser              5.1.0              pyhd8ed1ab_0    conda-forge
cookiecutter              1.6.0                 py38_1000    conda-forge
cryptography              3.4.7            py38hd7da0ea_0    conda-forge
cudatoolkit               11.3.1               h59b6b97_2
cupy                      9.5.0            py38hf95616d_1    conda-forge
cusignal                  21.08.00        py37_g33f663e_0    rapidsai
cycler                    0.10.0                     py_2    conda-forge
cython                    0.29.24          py38h885f38d_0    conda-forge
dash                      2.0.0              pyhd8ed1ab_0    conda-forge
dataclasses               0.8                pyhc8e2a94_3    conda-forge
debugpy                   1.4.1            py38h885f38d_0    conda-forge
decorator                 5.1.0              pyhd8ed1ab_0    conda-forge
defusedxml                0.7.1              pyhd8ed1ab_0    conda-forge
deprecated                1.2.13                   pypi_0    pypi
diff-match-patch          20200713           pyh9f0ad1d_0    conda-forge
docker-pycreds            0.4.0                      py_0    conda-forge
docutils                  0.17.1           py38haa244fe_0    conda-forge
editorconfig              0.12.3                   pypi_0    pypi
entrypoints               0.3             pyhd8ed1ab_1003    conda-forge
expat                     2.4.2                h39d44d4_0    conda-forge
fastrlock                 0.8              py38h885f38d_1    conda-forge
ffmpeg                    1.4                      pypi_0    pypi
fftw                      3.3.10          nompi_hea9a5d6_101    conda-forge
fire                      0.4.0              pyh44b312d_0    conda-forge
flake8                    4.0.1              pyhd8ed1ab_1    conda-forge
flask                     2.0.2              pyhd8ed1ab_0    conda-forge
flask-compress            1.10.1             pyhd8ed1ab_0    conda-forge
font-ttf-dejavu-sans-mono 2.37                 hab24e00_0    conda-forge
font-ttf-inconsolata      3.000                h77eed37_0    conda-forge
font-ttf-source-code-pro  2.038                h77eed37_0    conda-forge
font-ttf-ubuntu           0.83                 hab24e00_0    conda-forge
fontconfig                2.13.1            h1989441_1005    conda-forge
fonts-conda-ecosystem     1                             0    conda-forge
fonts-conda-forge         1                             0    conda-forge
freeglut                  3.2.1                h0e60522_2    conda-forge
freetype                  2.10.4               h546665d_1    conda-forge
fribidi                   1.0.10               h8d14728_0    conda-forge
fsspec                    2021.10.1          pyhd8ed1ab_0    conda-forge
future                    0.18.2           py38haa244fe_3    conda-forge
getopt-win32              0.1                  h8ffe710_0    conda-forge
gettext                   0.19.8.1          ha2e2712_1008    conda-forge
gin-config                0.5.0              pyhd8ed1ab_0    conda-forge
gitdb                     4.0.9              pyhd8ed1ab_0    conda-forge
gitpython                 3.1.24             pyhd8ed1ab_0    conda-forge
google-auth               1.35.0             pyh6c4a22f_0    conda-forge
google-auth-oauthlib      0.4.6              pyhd8ed1ab_0    conda-forge
graphite2                 1.3.13                     1000    conda-forge
graphviz                  2.50.0               hefbd956_1    conda-forge
grpcio                    1.41.1           py38he5377a8_1    conda-forge
gts                       0.7.6                h7c369d9_2    conda-forge
h5py                      3.6.0           nompi_py38hde0384b_100    conda-forge
harfbuzz                  3.0.0                hc601d6f_1    conda-forge
hdf5                      1.12.1          nompi_h2a0e4a3_103    conda-forge
icu                       68.1                 h0e60522_0    conda-forge
idna                      3.1                pyhd3deb0d_0    conda-forge
imageio                   2.13.5             pyh239f2a4_0    conda-forge
imagesize                 1.2.0                      py_0    conda-forge
importlib-metadata        4.2.0            py38haa244fe_0    conda-forge
importlib_metadata        4.2.0                hd8ed1ab_0    conda-forge
inflection                0.5.1              pyh9f0ad1d_0    conda-forge
iniconfig                 1.1.1              pyh9f0ad1d_0    conda-forge
intel-openmp              2021.3.0          h57928b3_3372    conda-forge
intervaltree              3.0.2                      py_0    conda-forge
ipykernel                 6.13.0           py38h4317176_0    conda-forge
ipython                   7.33.0           py38haa244fe_0    conda-forge
ipython_genutils          0.2.0                      py_1    conda-forge
isort                     5.9.3              pyhd8ed1ab_0    conda-forge
itsdangerous              2.0.1              pyhd8ed1ab_0    conda-forge
jams                      0.3.4                    pypi_0    pypi
jasper                    2.0.33               h77af90b_0    conda-forge
jbig                      2.1               h8d14728_2003    conda-forge
jedi                      0.18.0           py38haa244fe_2    conda-forge
jellyfish                 0.8.9            py38h294d835_2    conda-forge
jinja2                    3.0.2              pyhd8ed1ab_0    conda-forge
jinja2-time               0.2.0                      py_2    conda-forge
joblib                    1.1.0              pyhd8ed1ab_0    conda-forge
jpeg                      9d                   h8ffe710_0    conda-forge
jsbeautifier              1.14.0                   pypi_0    pypi
json5                     0.9.6              pyhd3eb1b0_0
jsonschema                4.1.0              pyhd8ed1ab_0    conda-forge
jupyter-console           6.4.0                    pypi_0    pypi
jupyter_client            7.1.2              pyhd8ed1ab_0    conda-forge
jupyter_core              4.8.1            py38haa244fe_0    conda-forge
jupyter_server            1.11.1             pyhd8ed1ab_0    conda-forge
jupyterlab                3.2.1              pyhd8ed1ab_0    conda-forge
jupyterlab-server         1.2.0                    pypi_0    pypi
jupyterlab_pygments       0.1.2              pyh9f0ad1d_0    conda-forge
jupyterlab_server         2.8.2              pyhd8ed1ab_0    conda-forge
keyring                   23.2.1           py38haa244fe_0    conda-forge
kiwisolver                1.3.2            py38hbd9d945_0    conda-forge
krb5                      1.19.2               h20d022d_3    conda-forge
lazy-object-proxy         1.6.0            py38h294d835_0    conda-forge
lcms2                     2.12                 h2a16943_0    conda-forge
lerc                      2.2.1                h0e60522_0    conda-forge
libarchive                3.5.2                hb45042f_1    conda-forge
libblas                   3.9.0              11_win64_mkl    conda-forge
libcblas                  3.9.0              11_win64_mkl    conda-forge
libclang                  11.1.0          default_h5c34c98_1    conda-forge
libcurl                   7.80.0               h789b8ee_1    conda-forge
libdeflate                1.7                  h8ffe710_5    conda-forge
libffi                    3.4.2                h8ffe710_5    conda-forge
libflac                   1.3.3                h0e60522_1    conda-forge
libgd                     2.3.3                h8bb91b0_0    conda-forge
libglib                   2.70.2               h3be07f2_1    conda-forge
libiconv                  1.16                 he774522_0    conda-forge
liblapack                 3.9.0              11_win64_mkl    conda-forge
liblapacke                3.9.0              11_win64_mkl    conda-forge
libmamba                  0.19.1               h44daa3b_0    conda-forge
libmambapy                0.19.1           py38h2bfd5b9_0    conda-forge
libogg                    1.3.5                h2bbff1b_1
libopencv                 4.5.3            py38hd5548a8_7    conda-forge
libopus                   1.3.1                h8ffe710_1    conda-forge
libpng                    1.6.37               h1d00b33_2    conda-forge
libprotobuf               3.19.1               h7755175_0    conda-forge
librosa                   0.8.1              pyhd8ed1ab_0    conda-forge
libsndfile                1.0.31               h0e60522_1    conda-forge
libsodium                 1.0.18               h8d14728_1    conda-forge
libsolv                   0.7.19               h7755175_5    conda-forge
libspatialindex           1.9.3                h39d44d4_4    conda-forge
libssh2                   1.10.0               h680486a_2    conda-forge
libtiff                   4.3.0                h0c97f57_1    conda-forge
libuv                     1.40.0               he774522_0
libvorbis                 1.3.7                ha925a31_0    conda-forge
libwebp                   1.2.1                h57928b3_0    conda-forge
libwebp-base              1.2.1                h8ffe710_0    conda-forge
libxcb                    1.13              hcd874cb_1004    conda-forge
libxml2                   2.9.12               hf5bbc77_1    conda-forge
libxslt                   1.1.34               he774522_0
libzlib                   1.2.11            h8ffe710_1013    conda-forge
llvmlite                  0.36.0           py38h57a6900_0    conda-forge
lxml                      4.8.0            py38h1985fb9_0
lz4-c                     1.9.3                h8ffe710_1    conda-forge
lzo                       2.10              hfa6e2cd_1000    conda-forge
m2w64-gcc-libgfortran     5.3.0                         6    conda-forge
m2w64-gcc-libs            5.3.0                         7    conda-forge
m2w64-gcc-libs-core       5.3.0                         7    conda-forge
m2w64-gmp                 6.1.0                         2    conda-forge
m2w64-libwinpthread-git   5.0.0.4634.697f757               2    conda-forge
mamba                     0.19.1           py38hecfeebb_0    conda-forge
markdown                  3.3.4              pyhd8ed1ab_0    conda-forge
markupsafe                2.0.1            py38h294d835_0    conda-forge
matplotlib                3.4.3            py38haa244fe_1    conda-forge
matplotlib-base           3.4.3            py38h1f000d6_1    conda-forge
matplotlib-inline         0.1.3              pyhd8ed1ab_0    conda-forge
mccabe                    0.6.1                      py_1    conda-forge
menuinst                  1.4.18           py38haa244fe_1    conda-forge
mido                      1.2.10                   pypi_0    pypi
mir-eval                  0.7                      pypi_0    pypi
mirdata                   0.3.6                    pypi_0    pypi
mistune                   0.8.4           py38h294d835_1004    conda-forge
mkl                       2021.3.0           hb70f87d_564    conda-forge
more-itertools            8.10.0             pyhd8ed1ab_0    conda-forge
mpmath                    1.2.1              pyhd8ed1ab_0    conda-forge
msys2-conda-epoch         20160418                      1    conda-forge
multidict                 5.2.0            py38h294d835_1    conda-forge
mypy_extensions           0.4.3            py38haa244fe_3    conda-forge
nbclassic                 0.3.2              pyhd8ed1ab_0    conda-forge
nbclient                  0.5.4              pyhd8ed1ab_0    conda-forge
nbconvert                 5.6.1                    pypi_0    pypi
nbformat                  5.1.3              pyhd8ed1ab_0    conda-forge
nest-asyncio              1.5.1              pyhd8ed1ab_0    conda-forge
ninja                     1.10.2               h6d14046_1
nnaudio                   0.3.1                    pypi_0    pypi
notebook                  6.4.4              pyha770c72_0    conda-forge
numba                     0.53.0           py38h5c177ec_0    conda-forge
numpy                     1.22.0           py38hcf66579_0    conda-forge
numpydoc                  1.1.0                      py_1    conda-forge
oauthlib                  3.1.1              pyhd8ed1ab_0    conda-forge
olefile                   0.46               pyh9f0ad1d_1    conda-forge
opencv                    4.5.3            py38haa244fe_7    conda-forge
openjpeg                  2.4.0                hb211442_1    conda-forge
openssl                   1.1.1o               h8ffe710_0    conda-forge
packaging                 21.0               pyhd8ed1ab_0    conda-forge
pandas                    1.3.5            py38h5d928e2_0    conda-forge
pandoc                    2.14.2               h8ffe710_0    conda-forge
pandocfilters             1.5.0              pyhd8ed1ab_0    conda-forge
pango                     1.48.10              h33e4779_2    conda-forge
paramiko                  2.7.2              pyh9f0ad1d_0    conda-forge
parso                     0.8.2              pyhd8ed1ab_0    conda-forge
pathspec                  0.9.0              pyhd8ed1ab_0    conda-forge
pathtools                 0.1.2                      py_1    conda-forge
pcre                      8.45                 h0e60522_0    conda-forge
pdfkit                    0.6.1                    pypi_0    pypi
pexpect                   4.8.0              pyh9f0ad1d_2    conda-forge
pickleshare               0.7.5                   py_1003    conda-forge
pillow                    9.0.1            py38hdc2b20a_0
pip                       21.2.4             pyhd8ed1ab_0    conda-forge
pixman                    0.40.0               h8ffe710_0    conda-forge
pkginfo                   1.8.2              pyhd8ed1ab_0    conda-forge
platformdirs              2.3.0              pyhd8ed1ab_0    conda-forge
plotly                    5.3.1                      py_0    plotly
pluggy                    1.0.0            py38haa244fe_1    conda-forge
pooch                     1.5.2              pyhd8ed1ab_0    conda-forge
poyo                      0.5.0                      py_0    conda-forge
pretty-midi               0.2.9                    pypi_0    pypi
prometheus_client         0.11.0             pyhd8ed1ab_0    conda-forge
promise                   2.3              py38haa244fe_5    conda-forge
prompt-toolkit            3.0.20             pyha770c72_0    conda-forge
protobuf                  3.19.1           py38h885f38d_1    conda-forge
psutil                    5.8.0            py38h294d835_1    conda-forge
pthread-stubs             0.4               hcd874cb_1001    conda-forge
ptyprocess                0.7.0              pyhd3deb0d_0    conda-forge
py                        1.10.0             pyhd3deb0d_0    conda-forge
py-lz4framed              0.14.0                   pypi_0    pypi
py-opencv                 4.5.3            py38h595d716_7    conda-forge
pyasn1                    0.4.8                      py_0    conda-forge
pyasn1-modules            0.2.8                      py_0
pyautogui                 0.9.53           py38haa244fe_0    conda-forge
pybind11-abi              4                    hd8ed1ab_3    conda-forge
pycodestyle               2.8.0              pyhd8ed1ab_0    conda-forge
pycosat                   0.6.3           py38h294d835_1009    conda-forge
pycparser                 2.20               pyh9f0ad1d_2    conda-forge
pydeprecate               0.3.1              pyhd8ed1ab_0    conda-forge
pydocstyle                6.1.1              pyhd8ed1ab_0    conda-forge
pyfftw                    0.12.0           py38h46b76f8_3    conda-forge
pyflakes                  2.4.0              pyhd8ed1ab_0    conda-forge
pygments                  2.10.0             pyhd8ed1ab_0    conda-forge
pyjwt                     2.3.0              pyhd8ed1ab_0    conda-forge
pylint                    2.7.2            py38haa244fe_0    conda-forge
pyls-spyder               0.4.0              pyhd8ed1ab_0    conda-forge
pymsgbox                  1.0.9              pyh9f0ad1d_0    conda-forge
pynacl                    1.4.0            py38h31c79cd_2    conda-forge
pyopenssl                 21.0.0             pyhd8ed1ab_0    conda-forge
pyparsing                 2.4.7              pyh9f0ad1d_0    conda-forge
pypiwin32                 223                      pypi_0    pypi
pyqt                      5.12.3           py38haa244fe_7    conda-forge
pyqt-impl                 5.12.3           py38h885f38d_7    conda-forge
pyqt5-sip                 4.19.18          py38h885f38d_7    conda-forge
pyqtchart                 5.12             py38h885f38d_7    conda-forge
pyqtwebengine             5.12.1           py38h885f38d_7    conda-forge
pyrsistent                0.17.3           py38h294d835_2    conda-forge
pyscreeze                 0.1.27             pyhd8ed1ab_0    conda-forge
pysocks                   1.7.1            py38haa244fe_3    conda-forge
pysoundfile               0.10.3.post1       pyhd3deb0d_0    conda-forge
pytest                    6.2.5            py38haa244fe_0    conda-forge
python                    3.8.12          h7840368_1_cpython    conda-forge
python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
python-graphviz           0.19.1                   pypi_0    pypi
python-kaleido            0.0.3                    py38_0    plotly
python-lsp-black          1.2.1              pyhd8ed1ab_0    conda-forge
python-lsp-jsonrpc        1.0.0              pyhd8ed1ab_0    conda-forge
python-lsp-server         1.4.1              pyhd8ed1ab_1    conda-forge
python_abi                3.8                      2_cp38    conda-forge
pytorch                   1.11.0          py3.8_cuda11.3_cudnn8_0    pytorch
pytorch-lightning         1.6.0              pyhd8ed1ab_0    conda-forge
pytorch-mutex             1.0                        cuda    pytorch
pytweening                1.0.3                      py_0    conda-forge
pytz                      2021.3             pyhd8ed1ab_0    conda-forge
pyu2f                     0.1.5              pyhd8ed1ab_0    conda-forge
pywin32                   301              py38h294d835_0    conda-forge
pywin32-ctypes            0.2.0           py38haa244fe_1003    conda-forge
pywinpty                  1.1.4            py38hd3f51b4_0    conda-forge
pyyaml                    5.4.1            py38h294d835_1    conda-forge
pyzmq                     22.3.0           py38h09162b1_0    conda-forge
qdarkstyle                3.0.2              pyhd8ed1ab_0    conda-forge
qstylizer                 0.2.1              pyhd8ed1ab_0    conda-forge
qt                        5.12.9               h5909a2a_4    conda-forge
qtawesome                 1.0.3              pyhd8ed1ab_0    conda-forge
qtconsole                 5.3.0              pyhd8ed1ab_0    conda-forge
qtconsole-base            5.3.0              pyhd8ed1ab_0    conda-forge
qtpy                      2.1.0              pyhd8ed1ab_0    conda-forge
readme_renderer           27.0               pyh9f0ad1d_0    conda-forge
regex                     2021.10.8        py38h294d835_0    conda-forge
reproc                    14.2.3               h8ffe710_0    conda-forge
reproc-cpp                14.2.3               h0e60522_0    conda-forge
requests                  2.26.0             pyhd8ed1ab_0    conda-forge
requests-oauthlib         1.3.0              pyh9f0ad1d_0    conda-forge
requests-toolbelt         0.9.1                      py_0    conda-forge
requests-unixsocket       0.2.0                      py_0    conda-forge
resampy                   0.2.2                      py_0    conda-forge
rfa-toolbox               1.4.2                    pypi_0    pypi
rfc3986                   2.0.0              pyhd8ed1ab_0    conda-forge
rope                      0.20.1             pyhd8ed1ab_0    conda-forge
rsa                       4.7.2              pyh44b312d_0    conda-forge
rtree                     0.9.7            py38h8b54edf_2    conda-forge
ruamel_yaml               0.15.100         py38h2bbff1b_0
scikit-learn              1.0              py38h8224a6f_1    conda-forge
scipy                     1.7.3            py38ha1292f7_0    conda-forge
send2trash                1.8.0              pyhd8ed1ab_0    conda-forge
sentry-sdk                1.5.0              pyhd8ed1ab_0    conda-forge
setproctitle              1.2.3            py38h294d835_0    conda-forge
setuptools                58.2.0           py38haa244fe_0    conda-forge
shortuuid                 1.0.8            py38haa244fe_0    conda-forge
six                       1.16.0             pyh6c4a22f_0    conda-forge
smart-open                5.2.1                    pypi_0    pypi
smmap                     3.0.5              pyh44b312d_0    conda-forge
sniffio                   1.2.0            py38haa244fe_1    conda-forge
snowballstemmer           2.1.0              pyhd8ed1ab_0    conda-forge
sortedcontainers          2.4.0              pyhd8ed1ab_0    conda-forge
sounddevice               0.4.3                    pypi_0    pypi
sphinx                    4.2.0              pyh6c4a22f_0    conda-forge
sphinxcontrib-applehelp   1.0.2                      py_0    conda-forge
sphinxcontrib-devhelp     1.0.2                      py_0    conda-forge
sphinxcontrib-htmlhelp    2.0.0              pyhd8ed1ab_0    conda-forge
sphinxcontrib-jsmath      1.0.1                      py_0    conda-forge
sphinxcontrib-qthelp      1.0.3                      py_0    conda-forge
sphinxcontrib-serializinghtml 1.1.5              pyhd8ed1ab_0    conda-forge
spyder                    5.3.0            py38haa244fe_0    conda-forge
spyder-kernels            2.3.0            py38haa244fe_0    conda-forge
spyder-unittest           0.5.0              pyhd8ed1ab_0    spyder-ide
sqlite                    3.36.0               h8ffe710_2    conda-forge
subprocess32              3.5.4                      py_1    conda-forge
sympy                     1.9              py38haa244fe_0    conda-forge
tbb                       2021.3.0             h2d74725_0    conda-forge
tenacity                  8.0.1            py38haa95532_0
tensorboard               2.6.0              pyhd8ed1ab_1    conda-forge
tensorboard-data-server   0.6.0            py38haa244fe_1    conda-forge
tensorboard-plugin-wit    1.8.0              pyh44b312d_0    conda-forge
termcolor                 1.1.0                      py_2    conda-forge
terminado                 0.12.1           py38haa244fe_0    conda-forge
testpath                  0.5.0              pyhd8ed1ab_0    conda-forge
textdistance              4.2.1              pyhd8ed1ab_0    conda-forge
threadpoolctl             3.0.0              pyh8a188c0_0    conda-forge
three-merge               0.1.1              pyh9f0ad1d_0    conda-forge
tinycss2                  1.1.0              pyhd8ed1ab_0    conda-forge
tk                        8.6.11               h8ffe710_1    conda-forge
toml                      0.10.2             pyhd8ed1ab_0    conda-forge
tomli                     1.2.1              pyhd8ed1ab_0    conda-forge
torchaudio                0.11.0               py38_cu113    pytorch
torchinfo                 1.5.4              pyhd8ed1ab_0    conda-forge
torchmetrics              0.8.0.dev0               pypi_0    pypi
torchsummary              1.5.1                    pypi_0    pypi
torchvision               0.12.0               py38_cu113    pytorch
torchviz                  0.0.2                    pypi_0    pypi
tornado                   6.1              py38h294d835_1    conda-forge
tqdm                      4.62.3             pyhd8ed1ab_0    conda-forge
traitlets                 4.3.3                    pypi_0    pypi
twine                     3.7.1              pyhd8ed1ab_0    conda-forge
typed-ast                 1.4.3            py38h294d835_0    conda-forge
typing-extensions         4.1.1                hd8ed1ab_0    conda-forge
typing_extensions         4.1.1              pyha770c72_0    conda-forge
ucrt                      10.0.20348.0         h57928b3_0    conda-forge
ujson                     4.2.0            py38h885f38d_0    conda-forge
urllib3                   1.26.7             pyhd8ed1ab_0    conda-forge
vc                        14.2                 hb210afc_5    conda-forge
vs2015_runtime            14.29.30037          h902a5da_5    conda-forge
wandb                     0.12.15            pyhd8ed1ab_0    conda-forge
watchdog                  2.1.6            py38haa244fe_0    conda-forge
wcwidth                   0.2.5              pyh9f0ad1d_2    conda-forge
webencodings              0.5.1                      py_1    conda-forge
websocket-client          0.58.0           py38haa95532_4
werkzeug                  2.0.1              pyhd8ed1ab_0    conda-forge
wheel                     0.37.0             pyhd8ed1ab_1    conda-forge
whichcraft                0.6.1                      py_0    conda-forge
win10toast                0.9                      pypi_0    pypi
win_inet_pton             1.1.0            py38haa244fe_2    conda-forge
winpty                    0.4.3                         4    conda-forge
wrapt                     1.12.1           py38h294d835_3    conda-forge
xorg-kbproto              1.0.7             hcd874cb_1002    conda-forge
xorg-libice               1.0.10               hcd874cb_0    conda-forge
xorg-libsm                1.2.3             hcd874cb_1000    conda-forge
xorg-libx11               1.7.2                hcd874cb_0    conda-forge
xorg-libxau               1.0.9                hcd874cb_0    conda-forge
xorg-libxdmcp             1.1.3                hcd874cb_0    conda-forge
xorg-libxext              1.3.4                hcd874cb_1    conda-forge
xorg-libxpm               3.5.13               hcd874cb_0    conda-forge
xorg-libxt                1.2.1                hcd874cb_2    conda-forge
xorg-xextproto            7.3.0             hcd874cb_1002    conda-forge
xorg-xproto               7.0.31            hcd874cb_1007    conda-forge
xz                        5.2.5                h62dcd97_1    conda-forge
yaml                      0.2.5                he774522_0    conda-forge
yaml-cpp                  0.6.3                ha925a31_4    conda-forge
yapf                      0.31.0             pyhd8ed1ab_0    conda-forge
yarl                      1.7.2            py38h294d835_1    conda-forge
yaspin                    2.1.0              pyhd8ed1ab_0    conda-forge
zeromq                    4.3.4                h0e60522_1    conda-forge
zipp                      3.6.0              pyhd8ed1ab_0    conda-forge
zlib                      1.2.11            h8ffe710_1013    conda-forge
zstd                      1.5.0                h6255e5f_0    conda-forge
