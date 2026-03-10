"""Document loaders for Knowledge module."""

from syrin.knowledge.loaders._directory import DirectoryLoader
from syrin.knowledge.loaders._github import GitHubLoader
from syrin.knowledge.loaders._json import JSONLoader
from syrin.knowledge.loaders._markdown import MarkdownLoader
from syrin.knowledge.loaders._pdf import PDFLoader
from syrin.knowledge.loaders._python import PythonLoader
from syrin.knowledge.loaders._text import RawTextLoader, TextLoader
from syrin.knowledge.loaders._url import URLLoader
from syrin.knowledge.loaders._yaml import YAMLLoader

__all__ = [
    "DirectoryLoader",
    "GitHubLoader",
    "JSONLoader",
    "MarkdownLoader",
    "PDFLoader",
    "PythonLoader",
    "RawTextLoader",
    "TextLoader",
    "URLLoader",
    "YAMLLoader",
]
