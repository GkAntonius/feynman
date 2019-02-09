This is the documentation directory.

To produce the documentation, issue
    make html

It will be produced in an external directory called ../../feynman-docs/html/.

To deploy the documentation online, move into the external
documentation directory and issue
    git add .
    git commit -m "Updated doc"
    git push origin gh-pages

