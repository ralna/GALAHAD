%% This is file `galahad_pandoc.cls',

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{galahad_pandoc}
              [1999/01/07 v1.4a
 Standard LaTeX document class]

\renewcommand{\medskip}{}
\renewcommand{\noindent}{}
\renewcommand{\hspace}[1]{}
\let\oldvspace\vspace
\newcommand{\@vspacestar}[1]{{\oldvspace*{#1}}}
\newcommand{\@vspacenostar}[1]{{\oldvspace{#1}}}
\renewcommand{\vspace}[1]{\@ifstar{}{}}

% set the release names

\newcommand{\libraryname}{GA\-LA\-H\-AD}
%\newcommand{\releasename}{GALAHAD Optimization Library version $2.\epsilon$}
\newcommand{\releasename}{GALAHAD Optimization Library version $3.3$}

\newcommand{\galahad}{\textsf{GALAHAD}}
\newcommand{\lancelot}{\textsf{LANCELOT}}
\newcommand{\lancelota}{\textsf{LANCELOT A}}
\newcommand{\lancelotb}{\textsf{LANCELOT B}}

% external styles

\usepackage{graphicx,fancyheadings,pslatex,bm,ifthen,url,varioref}
%\usepackage{verbatim}
\usepackage{fancyvrb}

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
\newcommand{\itte}[1]{\item[\tt #1$^{\ddagger}$]}
\newcommand{\ittf}[1]{\item[\tt #1]}
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

%\includegraphics[totalheight=40pt]{galahad.small.ps}\hfill\includegraphics[totalheight=40pt]{stfc.eps}\hspace{1mm}\includegraphics[totalheight=40pt]{epm.eps}\hspace{2mm}\includegraphics[totalheight=40pt]{unamur.ps}

%\includegraphics[totalheight=40pt]{galahad.small.ps}\hfill\includegraphics[totalheight=40pt]{ukristfc.eps}\hspace{4mm}\includegraphics[totalheight=40pt]{epm.eps}\hspace{2mm}\includegraphics[totalheight=40pt]{unamur.ps}

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

\newcommand{\galsummary}{\section{\ref{galsummary}. SUMMARY}
             \label{galsummary} \resetcounters}
\newcommand{\galhowto}{\section{\ref{galhowto}. HOW TO USE THE PACKAGE}
             \label{galhowto} \resetcounters}
\newcommand{\galgeneral}{\section{\ref{galgeneral}. GENERAL INFORMATION}
             \label{galgeneral} \resetcounters \begin{description}}
\newcommand{\galmethod}{\end{description}\section{\ref{galmethod}. METHOD}
             \label{galmethod} \resetcounters}
\newcommand{\galexample}{\section{\ref{galexample}. EXAMPLE OF USE}
             \label{galexample} \resetcounters}
\newcommand{\galexamples}{\section{\ref{galexample}. EXAMPLES OF USE}
             \label{galexample} \resetcounters}

% subsection names

\newcommand{\galterminology}{\subsection{\ref{galterminology}.
             Basic terminology}
             \label{galterminology}}
\newcommand{\galmatrix}{\subsection{\ref{galmatrix}. Matrix storage formats}
             \label{galmatrix}}
\newcommand{\galsymbols}{\subsection{\ref{galsymbols}. The GALAHAD symbols}
             \label{galsymbols}}
\newcommand{\galtype}{\subsection{\ref{galtype}. The derived data type}
             \label{galtype}}
\newcommand{\galopenmp}{\subsection{\ref{galopenmp}. OpenMP}
             \label{galopenmp}}
\newcommand{\galintkinds}{\subsection{\ref{galintkinds}. Integer kinds}
             \label{galintkinds}}
\newcommand{\galtypes}{\subsection{\ref{galtypes}. The derived data types}
             \label{galtypes}}
\newcommand{\galarguments}{\subsection{\ref{galarguments}.
             Argument lists and calling sequences}
             \label{galarguments}}
\newcommand{\galreverse}{\subsection{Reverse communication}
             \label{galreverse}}
\newcommand{\galerrors}{\subsection{Warning and error messages}
             \label{galerrors}}
\newcommand{\galinfo}{\subsection{Information printed}
             \label{galinfo}}
\newcommand{\galfeatures}{\subsection{Further features}
             \label{galfeatures}}
\newcommand{\galcontrolfeatures}{\subsection{Setting control parameters}
             \label{galcontrolfeatures}}
\newcommand{\galbasics}{\subsection{A brief introduction to terminology and scope}
             \label{galbasics}}
\newcommand{\galexternal}{\subsection{External packages}
             \label{galexternal}}

% subsubsection names

% attributes

\newcommand{\galattributes}{\medskip\noindent{\bf ATTRIBUTES --- }}

\newcommand{\galversions}[1]{{\bf Versions: }{{#1}}}
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
\newcommand{\realdp}{{\tt REAL} (double precision in $ ${\tt \fullpackagename\-\_\-double})}
\newcommand{\complex}{{\tt COMPLEX}}
\newcommand{\complexdp}{{\tt COMPLEX} (double precision complex in $ $ {\tt \fullpackagename\_\-double})}
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

\newcommand{\eqn}[2]{\begin{equation} \label{{#1}} #2 \;\;\; (#1)\end{equation}}
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
                     \mbox{minimize} \vspace*{-1mm} \\\\
                     \mbox{ $\scriptstyle #1 $ }
                     \end{array} \;} }
\renewcommand{\Re}{\hbox{I\hskip -1.5pt R}}
\newcommand{\smallRe}
     {\mbox{\raisebox{-0.8pt}{\footnotesize I\hskip -1.5pt R\hskip -0.5pt}}}
\newcommand{\tinyRe}
     {\mbox{\raisebox{-0.8pt}{\scriptsize I\hskip -1.5pt R\hskip -0.5pt}}}

\newcommand{\bigsum}{\displaystyle \sum}

% Fractions

\newcommand{\sfrac}[2]{{\frac{#1}{#2}}}
\newcommand{\half}{\sfrac{1}{2}}
\newcommand{\third}{\sfrac{1}{3}}
\newcommand{\quarter}{\sfrac{1}{4}}
\newcommand{\threehalves}{\sfrac{3}{2}}
\newcommand{\fivehalves}{\sfrac{5}{2}}

% super and subscripts

\newcommand{\s}[1]{^{\mbox{\protect\tiny #1}}}
\newcommand{\sub}[1]{_{\mbox{\protect\tiny #1}}}

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
\newcommand{\galerroptional}{-79}
\newcommand{\galerralive}{-82}

\endinput

%%
%% End of file `galahad.cls'.
