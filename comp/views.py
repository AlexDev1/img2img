from django.http import HttpResponseForbidden
from django.shortcuts import render

# Create your views here.


def MergeImageToMask(request):
    if request.user.is_superuser and not request.method.POST:
        pass
    else:
        return HttpResponseForbidden()
