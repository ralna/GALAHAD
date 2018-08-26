%
%% This is file `galahad.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% classes.dtx  (with options: `galahad')
%%
%% This is a generated file.
%%
%% Copyright 1993 1994 1995 1996 1997 1998 1999
%% The LaTeX3 Project and any individual authors listed elsewhere
%% in this file.
%%
%% This file is part of the LaTeX2e system.
%% ----------------------------------------
%%
%% It may be distributed under the terms of the LaTeX Project Public
%% License, as described in lppl.txt in the base LaTeX distribution.
%% Either version 1.0 or, at your option, any later version.
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{galahad}
              [1999/01/07 v1.4a
 Standard LaTeX document class]

% set the release names

\newcommand{\libraryname}{GA\-LA\-H\-AD}
%\newcommand{\releasename}{GALAHAD Optimization Library version $2.\epsilon$}
\newcommand{\releasename}{GALAHAD Optimization Library version $3.1$}

\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagefalse
\if@compatibility\else
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
%\ExecuteOptions{letterpaper,10pt,oneside,onecolumn,final}
\ExecuteOptions{a4paper,10pt,twoside,onecolumn,final}
\ProcessOptions
\input{size1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{2\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\sectionmark##1{%
      \markboth {\MakeUppercase{%
        \ifnum \c@secnumdepth >\z@
          \thesection\quad
        \fi
        ##1}}{}}%
    \def\subsectionmark##1{%
      \markright {%
        \ifnum \c@secnumdepth >\@ne
          \thesubsection\quad
        \fi
        ##1}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\markboth
    \def\sectionmark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
          \thesection\quad
        \fi
        ##1}}}}
\fi
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
    }
  \if@titlepage
  \newcommand\maketitle{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \vskip 60\p@
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 3em%
    {\large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
      \vskip 1.5em%
    {\large \@date \par}%       % Set date in \large size.
  \end{center}\par
  \@thanks
  \vfil\null
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\else
\newcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}
\fi
\setcounter{secnumdepth}{3}
\newcounter {part}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thesection {\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\part{%
   \if@noskipsec \leavevmode \fi
   \par
   \addvspace{4ex}%
   \@afterindentfalse
   \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >\m@ne
       \Large\bfseries \partname~\thepart
       \par\nobreak
     \fi
     \huge \bfseries #2%
     \markboth{}{}\par}%
    \nobreak
    \vskip 3ex
    \@afterheading}
\def\@spart#1{%
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \huge \bfseries #1\par}%
     \nobreak
     \vskip 3ex
     \@afterheading}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\large\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \listparindent1.0em
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\if@titlepage
  \newenvironment{abstract}{%
      \titlepage
      \null\vfil
      \@beginparpenalty\@lowpenalty
      \begin{center}%
        \bfseries \abstractname
        \@endparpenalty\@M
      \end{center}}%
     {\par\vfil\null\endtitlepage}
\else
  \newenvironment{abstract}{%
      \if@twocolumn
        \section*{\abstractname}%
      \else
        \small
        \begin{center}%
          {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
      \fi}
      {\if@twocolumn\else\endquotation\fi}
\fi
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\if@compatibility
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
\newcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\@Alph\c@section}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\renewcommand \theequation {\@arabic\c@equation}
\newcounter{figure}
\renewcommand \thefigure {\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{3}
\newcommand\tableofcontents{%
    \section*{\contentsname
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    \@starttoc{toc}%
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty\@secpenalty
    \addvspace{2.25em \@plus\p@}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
       \if@compatibility
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
      \fi
    \endgroup
  \fi}
\newcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \addpenalty\@secpenalty
    \addvspace{1.0em \@plus\p@}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
  \fi}
\newcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
\newcommand\listoffigures{%
    \section*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}}%
    \@starttoc{lof}%
    }
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \section*{\listtablename
      \@mkboth{%
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}%
    \@starttoc{lot}%
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\section*{\refname
        \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\section*{\indexname}]%
                \@mkboth{\MakeUppercase\indexname}%
                        {\MakeUppercase\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\refname{References}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\appendixname{Appendix}
\newcommand\abstractname{Abstract}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{plain}
\pagenumbering{arabic}
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi

\newcommand{\galahad}{\textsf{GALAHAD}}
\newcommand{\lancelot}{\textsf{LANCELOT}}
\newcommand{\lancelota}{\textsf{LANCELOT A}}
\newcommand{\lancelotb}{\textsf{LANCELOT B}}

\usepackage{graphicx,fancyheadings,pslatex,bm,fancyvrb,ifthen,url}

% Page sizes

\setlength\textheight {21.5cm}
%\setlength\textheight {17.5cm}
\setlength\textwidth {16.5 cm}
\setlength\oddsidemargin {-1.0mm}
\setlength\evensidemargin{-2.0mm}
\setlength\topmargin {-10.0mm}
\setlength\headheight {2.05cm}
\setlength\headsep {8.0mm}
%\setlength\footskip {1.1cm}

% Page style

\newcommand{\galogo}{\includegraphics[totalheight=10pt]{galahad.small.ps}}
\pagestyle{fancyplain}
\headrulewidth 0.5mm
\footrulewidth 0.5mm
\plainheadrulewidth 0.0mm
\plainfootrulewidth 0.5mm
\newcommand{\topa}{\fancyplain{}{\large \bf \packagename}}
\newcommand{\topb}{\fancyplain{}{\large \bf \libraryname}}
%\newcommand{\botl}{{\bf \large All use is subject to licence. See
%{\tt http://galahad.rl.ac.uk/galahad-www/cou.html} . \\
% For any commercial application, a separate license must be signed.}}
\newcommand{\botl}{{\bf \large All use is subject to the conditions of
the GNU Lesser General Public License version 3. \\
See {\tt http://galahad.rl.ac.uk/galahad-www/cou.html} for full details.}}

\newcommand{\bottr}{\packagename \, (\today) \,\,\,\, \thepage}
\newcommand{\bottl}{\thepage \,\,\,\, \packagename \, (\today)}

%\newcommand{\toptra}{{\Large \bf \packagename} \vspace*{2mm} \\ $\mbox{ }$ \\ $\mbox{ }$}
%\newcommand{\toplb}{{\Large \bf \libraryname} \vspace*{2mm} \\ \botl \vspace*{0.1mm}}
%\newcommand{\toptlb}{{\Large \bf \packagename}  \vspace*{2mm} \\ \botl \vspace*{0.1mm}}
%\newcommand{\toprb}{{\Large \bf \libraryname} \vspace*{2mm} \\ $\mbox{ }$ \\ $\mbox{ }$}

\newcommand{\toptra}{{\Large \bf \packagename}}
\newcommand{\toplb}{{\Large \bf \galogo \hspace*{1mm} \libraryname}}
\newcommand{\toptlb}{{\Large \bf \packagename}}
\newcommand{\toprb}{{\Large \bf \libraryname \hspace*{1mm} \galogo}}

\newcommand{\topaa}{\fancyplain{}{\toptra}}
\newcommand{\topba}{\fancyplain{}{\toplb}}
\newcommand{\topab}{\fancyplain{}{\toptlb}}
\newcommand{\topbb}{\fancyplain{}{\toprb}}

\newcommand{\bottra}{$\mbox{ }$ \\ $\mbox{ }$ \\ $\mbox{ }$ \\ \bottr}
\newcommand{\botlb}{\botl \\ $\mbox{ }$ \\ \libraryname}
\newcommand{\bottlb}{\botl \\ $\mbox{ }$ \\ \bottl}
\newcommand{\botrb}{$\mbox{ }$ \\ $\mbox{ }$ \\ $\mbox{ }$ \\ \libraryname}

\newcommand{\botaa}{\fancyplain{\bottra}{\bottra}}
\newcommand{\botba}{\fancyplain{\botlb}{\botlb}}
\newcommand{\botab}{\fancyplain{\bottlb}{\bottlb}}
\newcommand{\botbb}{\fancyplain{\botrb}{\botrb}}

\lhead[\topab]{\topba}
\rhead[\topbb]{\topaa}
\lfoot[\botab]{\botba}
\rfoot[\botbb]{\botaa}
\chead[\fancyplain{}{}]{\fancyplain{}{}}
\cfoot[\fancyplain{}{}]{\fancyplain{}{}}

%     New commands

\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\theequation}{\arabic{section}.\arabic{equation}}
\renewcommand{\thetable}{\arabic{section}.\arabic{table}}
\renewcommand{\thefigure}{\arabic{section}.\arabic{figure}}

%  Specifics

\newcommand{\itt}[1]{\item[\tt #1]}
\newcommand{\ittf}[1]{\item[\tt \parbox{7mm}{{#1}}]}
\newcommand{\ittt}[1]{\item[\tt #1]\hspace*{-2mm}}
\setcounter{secnumdepth}{3}

\newcommand{\subroutine}[1]{\smallskip{\setlength{\labelsep}{0cm}
\setlength{\leftmargini}{2cm}
\begin{itemize}\item[{\tt CAL}]{\tt L #1}\end{itemize}}\smallskip}

\newcommand{\function}[1]{\smallskip{\setlength{\labelsep}{0cm}
\setlength{\leftmargini}{1.5cm}
\begin{itemize}\item[]{\tt #1}\end{itemize}}\smallskip}

% front page header

\vspace*{-30mm}
\newcommand{\galheader}{
\thispagestyle{plain}
\begin{figure}[th]

%\includegraphics[totalheight=40pt]{galahad.small.ps}\hfill\includegraphics[totalheight=55pt]{stfc.eps}\hspace{1mm}\includegraphics[totalheight=55pt]{epm.eps}\hspace{2mm}\includegraphics[totalheight=55pt]{unamur.ps}

\includegraphics[totalheight=40pt]{galahad.small.ps}\hfill\includegraphics[totalheight=40pt]{stfc.eps}\hspace{1mm}\includegraphics[totalheight=40pt]{epm.eps}\hspace{2mm}\includegraphics[totalheight=40pt]{unamur.ps}

\end{figure}
\vspace*{-4mm}
\noindent
\rule{\textwidth}{0.5mm}
\vspace*{-2mm}

\noindent
{\huge \bf GALAHAD \hfill \packagename}
\vspace*{-1.0mm}

\noindent
\rule{\textwidth}{0.5mm}

\noindent
{USER DOCUMENTATION \hfill \releasename}
\vspace*{-1.0mm}

\noindent
\rule{\textwidth}{0.5mm}
\vspace*{-5.0mm}
}

% use the example program and results

\input{galahaddir}
\newcommand{\packagedir}{\galahaddir/src/\package}
\newcommand{\packageexample}{\packagedir/\package s.f90}
\newcommand{\packageresults}{\packagedir/\package ds.output}
\newcommand{\packagedata}{\packagedir/\package s.data}
\newcommand{\packageexampleb}{\packagedir/\package s2.f90}
\newcommand{\packageresultsb}{\packagedir/\package ds2.output}
\newcommand{\packagedatab}{\packagedir/\package s2.data}
\newcommand{\packageexamplec}{\packagedir/\package s3.f90}
\newcommand{\packageresultsc}{\packagedir/\package ds3.output}
\newcommand{\packagedatac}{\packagedir/\package s3.data}

\newcommand{\packagenewexample}{\packagedir/\package _spec.f90}
\newcommand{\packagenewresults}{\packagedir/\package _specd.output}
\newcommand{\packagenewdata}{\packagedir/\package _spec.data}
\newcommand{\packagenewexampleb}{\packagedir/\package _spec2.f90}
\newcommand{\packagenewresultsb}{\packagedir/\package _specd2.output}
\newcommand{\packagenewdatab}{\packagedir/\package _spec2.data}
\newcommand{\packagenewexamplec}{\packagedir/\package _sspec3.f90}
\newcommand{\packagenewresultsc}{\packagedir/\package _specd3.output}
\newcommand{\packagenewdatac}{\packagedir/\package _spec3.data}

% section names

\newcommand{\resetcounters}{\setcounter{equation}{0} \setcounter{figure}{0}
 \setcounter{table}{0}}
\newcommand{\galsummary}{\section{SUMMARY\label{galsummary}} \resetcounters}
\newcommand{\galhowto}{\section{HOW TO USE THE PACKAGE\label{galhowto}}
                        \resetcounters}
\newcommand{\galgeneral}{\section{GENERAL INFORMATION\label{galgeneral}}
                         \resetcounters \begin{description}}
\newcommand{\galmethod}{\end{description}\section{METHOD\label{galmethod}}
                        \resetcounters}
\newcommand{\galexample}{\section{EXAMPLE OF USE\label{galexample}}
                        \resetcounters}
\newcommand{\galexamples}{\section{EXAMPLES OF USE\label{galexample}}
                        \resetcounters}

% subsection names

\newcommand{\galterminology}{\subsection{Basic terminology\label{galterminology}}}
\newcommand{\galmatrix}{\subsection{Matrix storage formats\label{galmatrix}}}
\newcommand{\galsymbols}{\subsection{The GALAHAD symbols\label{galsymbols}}}
\newcommand{\galtype}{\subsection{The derived data type\label{galtype}}}
\newcommand{\galtypes}{\subsection{The derived data types\label{galtypes}}}
\newcommand{\galarguments}{\subsection{Argument lists and calling sequences\label{galarguments}}}
\newcommand{\galreverse}{\subsection{Reverse communication\label{galreverse}}}
\newcommand{\galerrors}{\subsection{Warning and error messages\label{galerrors}}}
\newcommand{\galinfo}{\subsection{Information printed\label{galinfo}}}
\newcommand{\galfeatures}{\subsection{Further features\label{galfeatures}}}
\newcommand{\galcontrolfeatures}{\subsection{Setting control parameters\label{galcontrolfeatures}}}
\newcommand{\galbasics}{\subsection{A brief introduction to terminology and scope\label{galbasics}}}
\newcommand{\galexternal}{\subsection{External packages\label{galexternal}}}

% attributes

\newcommand{\galattributes}{\medskip\noindent{\bf ATTRIBUTES --- }}
\newcommand{\galversions}{{\bf Versions: }}
\newcommand{\galuses}{{\bf Uses: }}
\newcommand{\galcalls}{{\bf Calls: }}
\newcommand{\galdate}{{\bf Date: }}
\newcommand{\galorigin}{{\bf Origin: }}
\newcommand{\gallanguage}{{\bf Language: }}
\newcommand{\galparallelism}{{\bf Parallelism: }}
\newcommand{\galremark}{{\bf Remark: }}

% description names

\newcommand{\galcommon}{\item[Use of common:]}
\newcommand{\galworkspace}{\item[Workspace:]}
\newcommand{\galroutines}{\item[Other routines called directly:]}
\newcommand{\galmodules}{\item[Other modules used directly:]}
\newcommand{\galio}{\item[Input/output:]}
\newcommand{\galrestrictions}{\item[Restrictions:]}
\newcommand{\galportability}{\item[Portability:]}
\newcommand{\galreference}{\vspace*{5mm} \noindent {\large {\bf Reference:} }}
\newcommand{\galreferences}{\vspace*{5mm} \noindent {\large {\bf References:} }}

% data types

\newcommand{\present}{{\tt PRESENT}}
\newcommand{\integer}{{\tt INTEGER}}
\newcommand{\shortinteger}{{\tt INTEGER(short)}}
\newcommand{\longinteger}{{\tt INTEGER(long)}}
\newcommand{\logical}{{\tt LOGICAL}}
\newcommand{\character}{{\tt CHARACTER}}
\newcommand{\real}{{\tt REAL}}
\newcommand{\realdp}{{\tt REAL} (double precision
 in {\tt \fullpackagename\-\_\-double})}
\newcommand{\complex}{{\tt COMPLEX}}
\newcommand{\complexdp}{{\tt COMPLEX} (double precision complex
 in {\tt \fullpackagename\_\-double})}
\newcommand{\optional}{{\tt OPTIONAL}}
\newcommand{\associated}{{\tt ASSOCIATED}}
\newcommand{\allocated}{{\tt ALLOCATED}}
\newcommand{\intentin}{{\tt INTENT(IN)}}
\newcommand{\intentinout}{{\tt INTENT(INOUT)}}
\newcommand{\intentout}{{\tt INTENT(OUT)}}
\newcommand{\true}{{\tt .TRUE.}}
\newcommand{\false}{{\tt .FALSE.}}
\newcommand{\restriction}{{\bf Restriction: }}
\newcommand{\restrictions}{{\bf Restrictions: }}

% Other useful definitions

\newcommand{\eqn}[2]{\begin{equation}\label{#1}{#2}\end{equation}}
\newcommand{\disp}[1]{\[{#1}\]}

\newcommand{\vect}[1]{\left(\begin{array}{c}#1\end{array}\right)}
\newcommand{\mat}[2]{\left(\begin{array}{#1}#2\end{array}\right)}
\newcommand{\matinv}[2]{\left(\begin{array}{#1}#2\end{array}\right)^{-1}}
\newcommand{\arr}[2]{\begin{array}{#1}#2\end{array}}
\newcommand{\eqdef}{\stackrel{\rm def}{=}}

\newcommand{\req}[1]{(\ref{#1})}
\newcommand{\rim}[1]{\;\;\; \mbox{#1} \;\;\;}
\newcommand{\tim}[1]{\;\; \mbox{#1} \;\;}

\newcommand{\tbtable}[1]{\begin{table}[tb]
                         \begin{center}
                         \begin{tabular}{#1} }
\newcommand{\bctable}[1]{\begin{table}[!htb]
                         \begin{center}
                         \begin{tabular}{#1} }
\newcommand{\pctable}[1]{\begin{table}[hp]
                         \begin{center}
                         \begin{tabular}{#1} }
\newcommand{\ectable}[1]{\end{tabular}
                         \caption{#1}
                         \end{center}
                         \end{table}}

\newcommand{\minin}[1]{ {\renewcommand{\arraystretch}{0.8}
                     \begin{array}[t]{c}
                     \mbox{minimize} \vspace*{-1mm} \\
                     \mbox{ $\scriptstyle #1 $ }
                     \end{array} \;} }
\renewcommand{\Re}{\hbox{I\hskip -1.5pt R}}
\newcommand{\smallRe}
     {\mbox{\raisebox{-0.8pt}{\footnotesize I\hskip -1.5pt R\hskip -0.5pt}}}
\newcommand{\tinyRe}
     {\mbox{\raisebox{-0.8pt}{\scriptsize I\hskip -1.5pt R\hskip -0.5pt}}}

\newcommand{\bigsum}{\displaystyle \sum}

% Fractions

\newcommand{\sfrac}[2]{{\scriptstyle \frac{#1}{#2}}}
\newcommand{\half}{\sfrac{1}{2}}
\newcommand{\third}{\sfrac{1}{3}}
\newcommand{\quarter}{\sfrac{1}{4}}
\newcommand{\threehalves}{\sfrac{3}{2}}
\newcommand{\fivehalves}{\sfrac{5}{2}}

% Bold mathematics
% syntax: \bm[A-Z,a-z]

\newcommand{\bmA}{{\mathbf{A}}} \newcommand{\bmB}{{\mathbf{B}}}
\newcommand{\bmC}{{\mathbf{C}}} \newcommand{\bmD}{{\mathbf{D}}}
\newcommand{\bmE}{{\mathbf{E}}} \newcommand{\bmF}{{\mathbf{F}}}
\newcommand{\bmG}{{\mathbf{G}}} \newcommand{\bmH}{{\mathbf{H}}}
\newcommand{\bmI}{{\mathbf{I}}} \newcommand{\bmJ}{{\mathbf{J}}}
\newcommand{\bmK}{{\mathbf{K}}} \newcommand{\bmL}{{\mathbf{L}}}
\newcommand{\bmM}{{\mathbf{M}}} \newcommand{\bmN}{{\mathbf{N}}}
\newcommand{\bmO}{{\mathbf{O}}} \newcommand{\bmP}{{\mathbf{P}}}
\newcommand{\bmQ}{{\mathbf{Q}}} \newcommand{\bmR}{{\mathbf{R}}}
\newcommand{\bmS}{{\mathbf{S}}} \newcommand{\bmT}{{\mathbf{T}}}
\newcommand{\bmU}{{\mathbf{U}}} \newcommand{\bmV}{{\mathbf{V}}}
\newcommand{\bmW}{{\mathbf{W}}} \newcommand{\bmX}{{\mathbf{X}}}
\newcommand{\bmY}{{\mathbf{Y}}} \newcommand{\bmZ}{{\mathbf{Z}}}

\newcommand{\bma}{{\mathbf{a}}} \newcommand{\bmb}{{\mathbf{b}}}
\newcommand{\bmc}{{\mathbf{c}}} \newcommand{\bmd}{{\mathbf{d}}}
\newcommand{\bme}{{\mathbf{e}}} \newcommand{\bmf}{{\mathbf{f}}}
\newcommand{\bmg}{{\mathbf{g}}} \newcommand{\bmh}{{\mathbf{h}}}
\newcommand{\bmi}{{\mathbf{i}}} \newcommand{\bmj}{{\mathbf{j}}}
\newcommand{\bmk}{{\mathbf{k}}} \newcommand{\bml}{{\mathbf{l}}}
\newcommand{\bmm}{{\mathbf{m}}} \newcommand{\bmn}{{\mathbf{n}}}
\newcommand{\bmo}{{\mathbf{o}}} \newcommand{\bmp}{{\mathbf{p}}}
\newcommand{\bmq}{{\mathbf{q}}} \newcommand{\bmr}{{\mathbf{r}}}
\newcommand{\bms}{{\mathbf{s}}} \newcommand{\bmt}{{\mathbf{t}}}
\newcommand{\bmu}{{\mathbf{u}}} \newcommand{\bmv}{{\mathbf{v}}}
\newcommand{\bmw}{{\mathbf{w}}} \newcommand{\bmx}{{\mathbf{x}}}
\newcommand{\bmy}{{\mathbf{y}}} \newcommand{\bmz}{{\mathbf{z}}}

\newcommand{\bmzero}{{\mathbf{0}}}
%\newcommand{\bmmu}{\mbox{\boldmath$\mu$}}
\newcommand{\bmmu}{\bm \mu}
\newcommand{\bmtheta}{\bm \theta}

% error codes (see symbols module)

\newcommand{\galerrallocate}{-1}
\newcommand{\galerrdeallocate}{-2}
\newcommand{\galerrrestrictions}{-3}
\newcommand{\galerrbadbounds}{-4}
\newcommand{\galerrprimalinfeasible}{-5}
\newcommand{\galerrdualinfeasible}{-6}
\newcommand{\galerrunbounded}{-7}
\newcommand{\galerrnocenter}{-8}
\newcommand{\galerranalysis}{-9}
\newcommand{\galerrfactorization}{-10}
\newcommand{\galerrsolve}{-11}
\newcommand{\galerrulsanalysis}{-12}
\newcommand{\galerrulsfactorization}{-13}
\newcommand{\galerrulssolve}{-14}
\newcommand{\galerrpreconditioner}{-15}
\newcommand{\galerrillconditioned}{-16}
\newcommand{\galerrtinystep}{-17}
\newcommand{\galerrmaxiterations}{-18}
\newcommand{\galwarnmaxiterations}{18}
\newcommand{\galerrcpulimit}{-19}
\newcommand{\galerrinertia}{-20}
\newcommand{\galerrfile}{-21}
\newcommand{\galerrio}{-22}
\newcommand{\galerrupperentry}{-23}
\newcommand{\galerrsort}{-24}
\newcommand{\galerrinput}{-25}
\newcommand{\galunknownsolver}{-26}
\newcommand{\galnotyetimplemented}{-27}
\newcommand{\galerrqpsolve}{-28}
\newcommand{\galerrunavailable}{-29}
\newcommand{\galwarnboundary}{-30}
\newcommand{\galerrcallorder}{-31}
\newcommand{\galerrintegerws}{-32}
\newcommand{\galerrrealws}{-33}
\newcommand{\galerrpardiso}{-34}
\newcommand{\galerrwsmp}{-35}
\newcommand{\galerrmcsixtyfour}{-36}
\newcommand{\galerrmcseventyseven}{-37}
\newcommand{\galerrlapack}{-38}
\newcommand{\galerrpermutation}{-39}
\newcommand{\galerralterdiagonal}{-40}
\newcommand{\galerraccesspivots}{-41}
\newcommand{\galerraccesspert}{-42}
\newcommand{\galerrdirectaccess}{-43}
\newcommand{\galerrfmin}{-44}
\newcommand{\galerrunknownprecond}{-45}
\newcommand{\galerrschurcomplement}{-46}
\newcommand{\galerrtechnical}{-50}
\newcommand{\galerrreformat}{-52}
\newcommand{\galerrahunordered}{-53}
\newcommand{\galerryunallocated}{-54}
\newcommand{\galerrzunallocated}{-55}
\newcommand{\galerrscale}{-61}
\newcommand{\galerrpresolve}{-62}
\newcommand{\galerrqpa}{-63}
\newcommand{\galerrqpb}{-64}
\newcommand{\galerrqpc}{-65}
\newcommand{\galerrcqp}{-66}
\newcommand{\galerrdqp}{-67}
\newcommand{\galerrmcsixtyone}{-69}
\newcommand{\galerrmcsictyeight}{-70}
\newcommand{\galerrmetis}{-71}
\newcommand{\galerrspral}{-72}
\newcommand{\galerrrepeated}{-73}
\newcommand{\galerreval}{-78}
\newcommand{\galerralive}{-82}

\endinput

%%
%% End of file `galahad.cls'.
