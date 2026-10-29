from release_common import freeze, server_check, archive
folder = freeze('portable.py', 'Curio', '--collect-all', 'curio', '--collect-all', 'transformers',
                '--collect-all', 'torch', '--collect-all', 'torchvision', '--collect-all', 'uvicorn',
                '--add-data', 'packs;packs', '--add-data', 'LICENSE;.', '--add-data', 'catalogs;catalogs')
server_check(folder / 'Curio.exe', 8765, '--smoke-test')
archive(folder, 'curio-windows.zip')
