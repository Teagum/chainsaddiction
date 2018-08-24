if &cp | set nocp | endif
nnoremap  h
let s:cpo_save=&cpo
set cpo&vim
nnoremap <NL> j
nnoremap  k
nnoremap  l
vmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
vnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(netrw#GX(),netrw#CheckIfRemote(netrw#GX()))
map <F6> :make && ./test_core
map <F5> :make
let &cpo=s:cpo_save
unlet s:cpo_save
set background=dark
set backspace=2
set fileencodings=ucs-bom,utf-8,default,latin1
set laststatus=2
set modelines=0
set runtimepath=~/.vim,~/.vim/pack/dist/start/vim-airline,~/.vim/pack/default/start/gruvbox,~/.vim/bundle/python-syntax,~/.vim/bundle/vim-colors-solarized,/usr/share/vim/vimfiles,/usr/share/vim/vim80,/usr/share/vim/vimfiles/after,~/.vim/after
set window=0
" vim: set ft=vim :
