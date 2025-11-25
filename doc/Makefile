
.PHONY=doc,install_doc_requirements,archive_doc

doc:
	sphinx-build -b html src html

install_doc_requirements:
	pip3 install -r requirements.txt

archive_doc:
	tar -cvzf html.tar.gz html/
