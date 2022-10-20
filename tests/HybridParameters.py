import collections

def ceilquotient(a,b):
  return -(a//-b)

def emptyList():
  return []

class ParameterCollection():
# Holds a collection of Params for a given program

  def __init__(self, fillParameters=emptyList, *optArgs):
    # fillParameters: A funtion that returns a list of Parameters.
    #
    # optArgs: Optional arguments for fillParameters. If None, then
    # fillParameters is called without arguments.

    self.optArgs=optArgs
    if optArgs != None:
      vals=fillParameters(*optArgs)
    else:
      vals=fillParameters()

    if type(vals) == list:
      for value in vals:
        if type(value) != Parameters:
          exit("fillOptValues must return a list of OptValues.")
      self.vals=vals
    else:
      exit("fillOptValues must return a list of OptValues.")

  def __eq__(self, other):
    if isinstance(other, ParameterCollection):
      return collections.Counter(self.vals) == collections.Counter(other.vals)
    return False

  # sortX() sorts self.val by the value of X
  def sortL(self, reverse=False):
    def byL(v):
      return v.L
    self.vals.sort(reverse=reverse,key=byL)

  def sortM(self, reverse=False):
    def f(v):
      return v.M
    self.vals.sort(reverse=reverse,key=f)

  def sortm(self, reverse=False):
    def f(v):
      return v.m
    self.vals.sort(reverse=reverse,key=f)

  def sortp(self, reverse=False):
    def f(v):
      return v.p
    self.vals.sort(reverse=reverse,key=f)

  def sortq(self, reverse=False):
    def f(v):
      return v.q
    self.vals.sort(reverse=reverse,key=f)

  def sortC(self, reverse=False):
    def f(v):
      return v.C
    self.vals.sort(reverse=reverse,key=f)

  def sortS(self, reverse=False):
    def f(v):
      return v.S
    self.vals.sort(reverse=reverse,key=f)

  def sortD(self, reverse=False):
    def f(v):
      return v.D
    self.vals.sort(reverse=reverse,key=f)

  def sortI(self, reverse=False):
    def f(v):
      return v.I
    self.vals.sort(reverse=reverse,key=f)

  def sortt(self, reverse=False):
    def f(v):
      return v.t
    self.vals.sort(reverse=reverse,key=f)

  # Returns a list of L values (respecting the order of self.val)
  def L(self):
    values=[]
    for v in self.vals:
      values.append(v.L)
    return values

  # Returns a list of M values (respecting the order of self.val)
  def M(self):
    values=[]
    for v in self.vals:
      values.append(v.M)
    return values

  # Returns a list of m values (respecting the order of self.val)
  def m(self):
    values=[]
    for v in self.vals:
      values.append(v.m)
    return values

  # Returns a list of p values (respecting the order of self.val)
  def p(self):
    values=[]
    for v in self.vals:
      values.append(v.p)
    return values

  # Returns a list of q values (respecting the order of self.val)
  def q(self):
    values=[]
    for v in self.vals:
      values.append(v.q)
    return values

  # Returns a list of C values (respecting the order of self.val)
  def C(self):
    values=[]
    for v in self.vals:
      values.append(v.C)
    return values

  # Returns a list of S values (respecting the order of self.val)
  def S(self):
    values=[]
    for v in self.vals:
      values.append(v.S)
    return values

  # Returns a list of D values (respecting the order of self.val)
  def D(self):
    values=[]
    for v in self.vals:
      values.append(v.D)
    return values

  # Returns a list of I values (respecting the order of self.val)
  def I(self):
    values=[]
    for v in self.vals:
      values.append(v.I)
    return values

  # Returns a list of I values (respecting the order of self.val)
  def t(self):
    values=[]
    for v in self.vals:
      values.append(v.t)
    return values

  # Returns a dictionary of lists of values (respecting the order of self.val)
  def allParams(self):
    params={}
    params["L"]=self.L()
    params["M"]=self.M()
    params["m"]=self.m()
    params["p"]=self.p()
    params["q"]=self.q()
    params["C"]=self.C()
    params["S"]=self.S()
    params["D"]=self.D()
    params["I"]=self.I()
    params["t"]=self.t()
    return params

class Parameters():
# A collection of parameters for a hybrid dealiased convolution
# Two Parameters objects are equal if all of their parameters OTHER THAN t are equal.
  def __init__(self,L,M,m,p,q,C,S,D,I,t=-1):
    self.L=int(L)
    self.M=int(M)
    self.m=int(m)
    self.p=int(p)
    self.q=int(q)
    self.C=int(C)
    self.S=int(S)
    self.D=int(D)
    self.I=int(I)
    self.t=float(t)

  def __hash__(self):
    return hash((self.L, self.M, self.m, self.p, self.q, self.C, self.S, self.D, self.I))

  def __eq__(self, other):
    if isinstance(other, Parameters):
      if self.L != other.L:
        return False
      if self.M != other.M:
        return False
      if self.m != other.m:
        return False
      if self.p != other.p:
        return False
      if self.q != other.q:
        return False
      if self.C != other.C:
        return False
      if self.S != other.S:
        return False
      if self.D != other.D:
        return False
      if self.I != other.I:
        return False
      return True
    return False