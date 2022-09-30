syntax on
colorscheme desert
set background=dark
set guifont=Monospace\ 11
hi normal guifg=white guibg=black
hi comment ctermfg=red guifg=red

imap jj <Esc>

set nocompatible
set confirm

set autoindent
set cindent
set smartindent
set tabstop=4
set softtabstop=2
set shiftwidth=4

set expandtab
set smarttab

set cino=g2,h2,N-s

set number
set history=100

set nobackup

set ignorecase
set smartcase

set hlsearch
set incsearch

set laststatus=2
set statusline=%F%m%r%h%w\[TYPE=%Y]\[POS=%l,%v][%p%%]\%{strftime(\"%d/%m/%y\ -\ %H:%M\")}

filetype on
filetype plugin on
filetype indent on

set report=0

set showmatch
set matchtime=5
set scrolloff=3

" set cursorline
" hi CursorLine term=bold cterm=bold gui=bold ctermbg=0x808080 guibg=grey4

set visualbell

autocmd BufNewFile,BufRead * setlocal formatoptions-=cro

autocmd BufRead,BufNewFile SConscript set filetype=python
autocmd BufRead,BufNewFile *.json set filetype=json
autocmd BufNewFile,BufRead *.py set filetype=python
autocmd BufNewFile,BufRead *.m set filetype=octave
autocmd BufNewFile,BufRead *.cu set filetype=cuda
autocmd BufNewFile,BufRead *.cl set filetype=opencl
autocmd BufNewFile,BufRead makefile.rules set filetype=make

fun! ShowFuncName()
    let lnum = line(".")
    let col = col(".")
    echohl ModeMsg
    echo getline(search("^[^ \t#/]\\{2}.*[^:]\s*$", 'bW'))
    echohl None
    call search("\\%" . lnum . "l" . "\\%" . col . "c")
endfun
map f :call ShowFuncName() <CR>
