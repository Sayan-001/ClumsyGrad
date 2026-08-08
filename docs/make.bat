@ECHO OFF

REM Command file for building ClumsyGrad documentation

pushd %~dp0

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help
if "%1" == "html" goto html
if "%1" == "pdf" goto pdf
if "%1" == "clean" goto clean
goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
    echo.installed, then set the SPHINXBUILD environment variable to point
    echo.to the full path of the 'sphinx-build' executable. Alternatively you
    echo.may add the Sphinx directory to PATH.
    echo.
    echo.If you don't have Sphinx installed, grab it from
    echo.http://sphinx-doc.org/
    exit /b 1
)

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%\html %SPHINXOPTS%
goto end

:pdf
%SPHINXBUILD% -b latex %SOURCEDIR% %BUILDDIR%\latex %SPHINXOPTS%
cd %BUILDDIR%\latex
pdflatex -interaction=nonstopmode ClumsyGrad.tex
pdflatex -interaction=nonstopmode ClumsyGrad.tex
cd %~dp0
goto end

:clean
rmdir /s /q %BUILDDIR%
goto end

:help
echo.Available targets:
echo.  html   Build HTML documentation into %BUILDDIR%\html
echo.  pdf    Build PDF documentation into %BUILDDIR%\latex
echo.  clean  Remove the build directory

:end
popd
