
.PHONY=doc,update_theme,archive_doc

doc:
	pip3 install sphinx==6.2 --upgrade
	sphinx-build -b html src html

archive_doc:
	tar -cvzf html.tar.gz html/
