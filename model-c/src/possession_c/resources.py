"""Bound local CPU inference and leave CUDA unused before importing Torch."""
import os


def restrict_resources():
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[name]='2'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    result=dict(threads=2,gpu_used=False)
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetCurrentProcess.restype=wintypes.HANDLE
        k.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        k.SetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
        k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        k.GetPriorityClass.argtypes=[wintypes.HANDLE]
        handle=k.GetCurrentProcess();current=ctypes.c_size_t();system=ctypes.c_size_t()
        if not k.GetProcessAffinityMask(handle,ctypes.byref(current),ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        cores=[i for i in range(current.value.bit_length()) if current.value&(1<<i)]
        mask=sum(1<<i for i in cores[-2:])
        if not mask or not k.SetProcessAffinityMask(handle,mask) or not k.SetPriorityClass(handle,0x4000):
            raise RuntimeError('Cannot apply the CPU affinity and below-normal priority')
        checked=ctypes.c_size_t()
        if not k.GetProcessAffinityMask(handle,ctypes.byref(checked),ctypes.byref(system)) or checked.value!=mask or k.GetPriorityClass(handle)!=0x4000:
            raise RuntimeError('Resource limits were not applied')
        result.update(affinity=[i for i in cores if mask&(1<<i)],priority='BELOW_NORMAL')
    elif hasattr(os,'sched_getaffinity'):
        cpus=sorted(os.sched_getaffinity(0))[-2:]
        os.sched_setaffinity(0,cpus)
        os.nice(5)
        result.update(affinity=cpus,priority='nice +5')
    return result
