# Configuration file for the Sphinx documentation builder.

import os
import sys

# 2 levels from docs/source to reach project root
sys.path.insert(0, os.path.abspath('../../'))
# Adding the src directory specifically
sys.path.insert(0, os.path.abspath('../../src/'))

# -- Project information -----------------------------------------------------
project = 'ClumsyGrad'
copyright = '2025, Sayan Gupta'
author = 'Sayan Gupta'
release = '0.1.0'
version = '0.1.0'

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
latex_engine = 'pdflatex' # Specifies the LaTeX engine to use

# Enhanced LaTeX configuration for beautiful PDF
latex_elements = {
    # --- Paper and Font Size ---
    'papersize': 'a4paper', # Paper size (a4paper or letterpaper)
    'pointsize': '11pt',    # Base font size (10pt, 11pt, or 12pt)

    # --- LaTeX Preamble ---
    # Custom LaTeX commands added to the beginning of the .tex file
    'preamble': r'''
        % --- Essential Packages ---
        \usepackage[utf8]{inputenc}     % Ensure UTF-8 input encoding
        \usepackage[T1]{fontenc}        % Use T1 font encoding for better character support and hyphenation

        % --- Font Selection (Example: Latin Modern for a clean look) ---
        \usepackage{lmodern}            % Provides the Latin Modern font family (a good default)
        \usepackage{inconsolata}        % A good monospace font for code listings

        \usepackage{microtype}          % Improves justification and visual appearance of text
        \setlength{\parindent}{0pt}      % No indentation for new paragraphs
        \setlength{\parskip}{0.5em}      % Space between paragraphs

        % --- Color Definitions ---
        \usepackage{xcolor}
        \definecolor{sphinxblue}{RGB}{0,102,204}       % A blue color for headings and links
        \definecolor{sphinxtitle}{RGB}{47,68,78}      % A darker color for main titles
        \definecolor{sphinxcodebg}{RGB}{248,249,250}  % Light background for code blocks

        % --- Chapter and Section Styling (using titlesec) ---
        % This provides a clean, modern look for headings.
        \usepackage{titlesec}
        \titleformat{\chapter}[display]
          {\normalfont\huge\bfseries\color{sphinxtitle}} % Chapter title format
          {\chaptertitlename\ \thechapter}{20pt}{\Huge\color{sphinxblue}} % Chapter name, number, spacing, style
        \titleformat{\section}
          {\normalfont\Large\bfseries\color{sphinxblue}} % Section title format
          {\thesection}{1em}{}                          % Section number, spacing, style
        \titleformat{\subsection}
          {\normalfont\large\bfseries\color{sphinxblue}} % Subsection title format
          {\thesubsection}{1em}{}                       % Subsection number, spacing, style
        \titleformat{\subsubsection}
          {\normalfont\normalsize\bfseries\color{sphinxblue}} % Subsubsection title format
          {\thesubsubsection}{1em}{}                     % Subsubsection number, spacing, style

        % --- List Styling (more compact lists) ---
        \usepackage{enumitem}
        \setlist{nosep} % Removes extra vertical spacing in lists

        % --- Code Block Styling (Sphinx handles most of this via Pygments) ---
        % The 'sphinxsetup' key can be used for Sphinx-specific Verbatim environments
        % \sphinxsetup{verbatimhintsturnover=true, verbatimwrapslines=true}

        % --- Hyperlinks (hyperref is loaded by Sphinx) ---
        % These settings customize the appearance of links in the PDF
        \hypersetup{
            colorlinks=true,                % Links are colored (not boxed)
            linkcolor=sphinxblue,           % Color for internal links
            urlcolor=sphinxblue,            % Color for external URLs
            citecolor=sphinxblue,           % Color for citation links
            bookmarks=true,                 % Create PDF bookmarks
            bookmarksopen=true,             % Open bookmarks tree by default
            bookmarksopenlevel=2,           % Expand bookmarks to this level
            pdfpagemode=UseOutlines,        % Show bookmarks panel when PDF opens
            pdftitle={\@title},             % Use the document title for PDF metadata
            pdfauthor={\@author},           % Use the document author for PDF metadata
            pdfsubject={ClumsyGrad Library Documentation}, % PDF subject
            pdfkeywords={Python, NumPy, Autodiff, ClumsyGrad} % PDF keywords
        }

        % --- Table of Contents Styling ---
        % Sphinx generates the \tableofcontents. These lines can fine-tune its appearance.
        \usepackage{tocloft}
        \renewcommand{\cftchapfont}{\normalfont\Large\bfseries} % Chapter font in ToC
        \renewcommand{\cftsecfont}{\normalfont\bfseries}        % Section font in ToC
        \setlength{\cftbeforechapskip}{0.5em}                   % Space before chapter entries in ToC
        \setlength{\cftbeforesecskip}{0.2em}                    % Space before section entries in ToC

        % --- Header and Footer (Example: Page numbers in footer) ---
        \usepackage{fancyhdr}           % Package for custom headers and footers
        \pagestyle{fancy}               % Apply fancy page style
        \fancyhf{}                      % Clear existing header/footer
        \fancyfoot[C]{\thepage}         % Page number centered in footer
        \renewcommand{\headrulewidth}{0pt} % No horizontal line in header
        \renewcommand{\footrulewidth}{0pt} % No horizontal line in footer
        % Apply plain style to chapter pages (no header/footer, just page number)
        \fancypagestyle{plain}{%
            \fancyhf{}%
            \fancyfoot[C]{\thepage}%
            \renewcommand{\headrulewidth}{0pt}%
            \renewcommand{\footrulewidth}{0pt}%
        }

        % --- MakeTitle (Title Page) ---
        % Sphinx generates a title page based on latex_documents.
        % The 'maketitle' element can be used for a very custom one.
        % For simplicity, we'll let Sphinx's default work, which is often sufficient.
        % If you need a custom title page, you can redefine \maketitle here.
        % Example of a simpler custom title:
        % \renewcommand{\maketitle}{%
        %   \begin{titlepage}%
        %     \centering%
        %     \vspace*{\stretch{1}}%
        %     {\Huge\bfseries\color{sphinxtitle} \@title \par}%
        %     \vspace{0.5cm}%
        %     {\Large\itshape \@author \par}%
        %     \vspace*{\stretch{2}}%
        %     {\large \today \par}%
        %   \end{titlepage}%
        %   \cleardoublepage%
        % }

        % --- Other useful packages ---
        \usepackage{emptypage}          % Ensures truly blank pages are empty (no headers/footers)
        \usepackage{amsmath, amssymb}   % For advanced math typesetting (if not already loaded by mathjax)
    ''',

    'figure_align': 'H', # Try to place figures 'Here' exactly, if possible

    # --- Class Options ---
    # 'openany': Allows chapters to start on any page (left or right), reducing blank pages.
    # 'oneside': For documents not intended for two-sided printing (simpler layout).
    'extraclassoptions': 'openany,oneside',
}

# --- LaTeX Document Configuration ---
# This defines the main .tex file and its properties.
latex_documents = [
    (master_doc,                # Source start file (index.rst)
     'ClumsyGrad.tex',          # Output .tex file name
     project,                   # PDF document title (uses 'project' variable)
     author,                    # PDF document author (uses 'author' variable)
     'manual',                  # Document class (manual or howto)
     True                       # Add a separate table of contents page
    ),
]

# --- Other LaTeX Options ---
latex_toplevel_sectioning = 'chapter' # Top-level sections in .rst become chapters in LaTeX
# latex_show_urls = 'footnote'          # Display URLs as footnotes (or 'inline' or 'no')
# latex_use_xindy = False             # Set to True if you use Xindy for indexing (requires Xindy installed)

# --- Numbering for Figures, Tables, Code Blocks ---
numfig = True                         # Enable automatic numbering
numfig_secnum_depth = 2               # Numbering depth (e.g., Figure 1.1, Figure 1.1.1)
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
}