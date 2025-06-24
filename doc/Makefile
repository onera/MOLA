
.PHONY=doc,update_theme,archive_doc

doc:
	cd src && python3 conf.py -r 

update_theme: 
	pip3 install https://gitlab.onera.net/onera/onera-sphinx-theme/-/raw/master/versions/onera_sphinx_theme-0.0.1-py3-none-any.whl --target theme --upgrade --no-deps
    
install_theme:
	pip3 install sphinx --target theme --upgrade
	pip3 install https://gitlab.onera.net/onera/onera-sphinx-theme/-/raw/master/versions/onera_sphinx_theme-0.0.1-py3-none-any.whl --target theme --upgrade

archive_doc:
	tar -cvzf html.tar.gz html/
