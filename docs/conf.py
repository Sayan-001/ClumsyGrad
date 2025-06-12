# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the source code to the Python path
# Go up 2 levels from docs/source to reach project root
sys.path.insert(0, os.path.abspath('../'))
# Add the src directory specifically
sys.path.insert(0, os.path.abspath('../src/'))

# -- Project information -----------------------------------------------------
project = 'ClumsyGrad'
copyright = '2025, Sayan Gupta'
author = 'Sayan Gupta'
release = '0.0.1'
version = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',          # Automatic documentation from docstrings
    'sphinx.ext.viewcode',         # Add source code links
    'sphinx.ext.napoleon',         # Support for Google/NumPy style docstrings
    'sphinx.ext.intersphinx',      # Link to other documentation
    'sphinx.ext.mathjax',          # Math support
    'sphinx.ext.githubpages',      # GitHub Pages support
    'sphinx.ext.todo',             # Todo extension
    'sphinx.ext.ifconfig',         # Conditional content
]

mathjax_config = {
    'extensions': ['tex2jax.js'],
    'jax': ['input/TeX', 'output/HTML-CSS'],
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_preprocess_types = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'

html_theme_options = {
    "repository_url": "https://github.com/Sayan-001/ClumsyGrad",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "show_navbar_depth": 1,
    "logo": {
        "text": "ClumsyGrad",
    },
    "extra_footer": "<p>Built with ❤️ for educational purposes</p>",
    "search_bar_text": "Search the docs...",
}

html_static_path = ['_static']
html_title = f"{project} v{version}"

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Fix for your import structure
autodoc_mock_imports = []

# -- Options for LaTeX output ------------------------------------------------
latex_engine = 'pdflatex'

# Enhanced LaTeX configuration for beautiful PDF
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'a4paper',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '11pt',

    # Additional stuff for the LaTeX preamble.
    'preamble': r'''
        % Enhanced typography
        \usepackage{charter}                    % Main text font
        \usepackage[defaultsans]{lato}          % Sans-serif font
        \usepackage{inconsolata}                % Monospace font
        \usepackage{microtype}                  % Better typography
        
        % Enhanced colors
        \usepackage{xcolor}
        \definecolor{titlecolor}{RGB}{47, 68, 78}
        \definecolor{chaptercolor}{RGB}{0, 102, 204}
        \definecolor{codebackground}{RGB}{248, 249, 250}
        \definecolor{codeborder}{RGB}{229, 231, 235}
        
        % Enhanced chapter/section styling
        \usepackage{titlesec}
        \titleformat{\chapter}[display]
            {\normalfont\huge\bfseries\color{chaptercolor}}
            {\chaptertitlename\ \thechapter}{20pt}{\Huge}
        \titleformat{\section}
            {\normalfont\Large\bfseries\color{titlecolor}}
            {\thesection}{1em}{}
        \titleformat{\subsection}
            {\normalfont\large\bfseries\color{titlecolor}}
            {\thesubsection}{1em}{}
            
        % Enhanced table styling
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{array}
        
        % Better list styling
        \usepackage{enumitem}
        \setlist{noitemsep, topsep=0.5em}
        
        % Enhanced code blocks
        \usepackage{fancyvrb}
        \usepackage{framed}
        
        % Custom title page
        \makeatletter
        \renewcommand{\maketitle}{%
            \begin{titlepage}
                \centering
                \vspace*{2cm}
                {\Huge\bfseries\color{titlecolor} ClumsyGrad}\\[0.5cm]
                {\Large A Simple Automatic Differentiation Library}\\[2cm]
                {\large Educational Deep Learning Framework}\\[1cm]
                {\large Built with NumPy for Learning Purposes}\\[3cm]
                {\Large\bfseries Documentation}\\[0.5cm]
                {\large Version \version}\\[2cm]
                \vfill
                {\large \author}\\[0.5cm]
                {\today}
            \end{titlepage}
        }
        \makeatother
        
        % Enhanced spacing
        \setlength{\parskip}{0.5em}
        \setlength{\parindent}{0pt}
        
        % Better footnotes
        \usepackage[bottom]{footmisc}
        
        % Enhanced hyperlinks
        \hypersetup{
            colorlinks=true,
            linkcolor=chaptercolor,
            urlcolor=chaptercolor,
            citecolor=chaptercolor,
            filecolor=chaptercolor,
            menucolor=chaptercolor,
            runcolor=chaptercolor,
            bookmarks=true,
            bookmarksopen=true,
            bookmarksopenlevel=2,
            pdfpagemode=UseOutlines,
            pdftitle={ClumsyGrad Documentation},
            pdfauthor={Sayan Gupta},
            pdfsubject={Automatic Differentiation Library},
            pdfkeywords={Deep Learning, Neural Networks, Automatic Differentiation, NumPy}
        }
    ''',

    # Latex figure (float) alignment
    'figure_align': 'H',
    
    # Remove blank pages and improve layout
    'extraclassoptions': 'openany,oneside',
    
    # Enhanced margins
    'geometry': r'\usepackage[top=2.5cm,bottom=2.5cm,left=3cm,right=2.5cm]{geometry}',
    
    # Footer customization
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, 'ClumsyGrad.tex', 'ClumsyGrad Documentation - A Simple Automatic Differentiation Library',
     'Sayan Gupta', 'manual'),
]

# LaTeX additional settings
latex_use_latex_multicolumn = True
latex_use_xindy = False

# Number figures, tables and code listings
numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s'
}

# Better section numbering
latex_toplevel_sectioning = 'chapter'