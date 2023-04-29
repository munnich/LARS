module Syllables
using DSP
using ArgParse
using CSV, DataFrames, FileIO, LibSndFile, SampledSignals
using Plots

intify(f) = trunc(Int, f)

"""
    preemphasis(data, pcoeff=-0.97)

Pre-emphasis processing with pre-emphasis coefficient `pcoeff`.
"""
function preemphasis(data, pcoeff=-0.97)
    DSP.filt([1, pcoeff], 1, data)
end

"""
    normalize(data)

Normalize array `data` by dividing it by its maximum magnitude value.
"""
function normalize(data)
    data / maximum(abs.(data))
end

"""
    cut_samples(data, framelen, overlap)

Cut down data array into samples of given frame length and given overlap with next sample.
If the array cannot be perfectly split into evenly sized samples, extra data at the end is thrown out.
"""
function cut_samples(data, framelen, overlap)
    # determine the number of samples we can fit
    n_samples = intify(floor(length(data) / framelen))
    # for this script it seems pretty important to maintain the framelen + overlap sample size
    # I can't think of an easier solution than this right now
    if length(data) < n_samples * framelen + overlap
        n_samples -= 1
    end
    samples = [data[(i - 1) * framelen + 1:i * framelen + overlap] for i in 1:n_samples]
    # array of arrays → matrix
    vcat(transpose.(samples)...)
end

"""
    ste(audio, framelen, inc)

Calculate short term energy (energy produced by the vocal cord vibration).
"""
function ste(audio; framelen=250, inc=100)
    sum(cut_samples(audio, framelen, inc) .^ 2; dims=2)
end

"""
    zcr(audio, framelen, inc)

Compute zero crossing rate.
"""
function zcr(audio; framelen=250, inc=100)
    tmp1 = cut_samples(audio[1:end-1], framelen, inc)
    tmp2 = cut_samples(audio[2:end], framelen, inc)
    sgn = @. (tmp1 * tmp2) < 0
    sum(sgn; dims=2)
end

"""
    maxthresh(data)

Calculate maximum ZCR and STE thresholds.
"""
function maxthresh(data)
    maximum(data) / 4
end

"""
    noise_analyze(syllablelen, framelen, overlap, fs, threshold=0.05)

Calculate whether a syllable sample can be considered as noise.
Raising the threshold will increase the number samples detected as noise.
"""
function noise_analyze(syllablelen, framelen, overlap, fs, threshold=0.05)
    if syllablelen * (framelen - overlap) / fs >= threshold
        return true
    end
    return false
end

"""
    syllable_length(STE, ZCR, thr, n, N, framelen=250, overlap=100, fs=44100)

Calculate syllable length and next sample to be analyzed using STE and ZCR arrays,
threshold dictionary, current sample number, and total sample number.
"""
function syllable_length(STE, ZCR, thr, n, N, framelen=250, overlap=100, fs=44100)
    init_n = n
    syllablelen = 0
    # check if syllable start
    if STE[n] > thr["ste_max"] || ZCR[n] > thr["zcr_max"]
        n += 1
        # run until we find a frame below the speech thresholds
        # I would've expected it to check for consonant - vowel switches not this
        while STE[n] > thr["ste_min"] && ZCR[n] > thr["zcr_min"] && n < N
            n += 1
        end
        # check if syllable is long enough to not be considered noise
        syllablelen = n - init_n
        if !noise_analyze(syllablelen, framelen, overlap, fs)
            syllablelen = 0
        end
    else
        n += 1
    end
    syllablelen, n
end

"""
    syllable_detection(stes, zcrs, thr; framelen=250, overlap=100, fs=44100)

Detect syllables using STEs, ZCRs, and min/max thresholds dictionary.
"""
function syllable_detection(stes, zcrs, thr; framelen=250, overlap=100, fs=44100)
    starts = Vector{Int}()
    ends = Vector{Int}()
    N = length(zcrs)
    n = 1
    while n < N
        old_n = n
        syllablelen, n = syllable_length(stes, zcrs, thr, n, N, framelen, overlap, fs)
        if syllablelen > 0
            push!(starts, old_n * framelen)
            push!(ends, n * framelen)
        end
    end
    if length(starts) == 0 
        return [0], [0]
    end
    starts, ends
end

"""
    getsyllables(audio, fs, framelen=150, overlap=100)

Calculate syllables according to  J. Xu, W. Liao and T. Inoue, "Speech Speed Awareness System for a Non-native Speaker," 2016 International Conference on Collaboration Technologies and Systems (CTS), Orlando, FL, USA, 2016, pp. 43-50, doi: 10.1109/CTS.2016.0027.
"""
function getsyllables(audio, fs, framelen=150, overlap=100)
    if size(audio, 2) == 2 
        audio = @. (audio[:, 1] + audio[:, 2]) / 2
        # audio = [(audio[i, 1] + audio[i, 2]) / 2 for i in 1:size(audio, 1)]
    end
    audio = preemphasis(audio)
    audio = normalize(audio)
    s = Threads.@spawn ste(audio, inc=overlap, framelen=framelen)
    z = Threads.@spawn zcr(audio, inc=overlap, framelen=framelen)
    stes = fetch(s)
    zcrs = fetch(z)
    p = []
    x = [i * framelen for i in 1:floor(length(audio) / framelen) - 1]
    push!(p, plot(audio, title="Audio", legend=false))
    push!(p, plot(x, stes, title="STE", legend=false))
    push!(p, plot(x, zcrs, title="ZCR", legend=false))
    
    savefig(plot(p..., layout=(3, 1), link=:x), "plot.png")
    thr = Dict(
              "ste_min" => minimum(stes) * 2,
              "zcr_min" => minimum(zcrs) * 2,
              "ste_max" => maxthresh(stes),
              "zcr_max" => maxthresh(zcrs)
              )
    syllable_detection(stes, zcrs, thr, fs=fs, overlap=overlap, framelen=framelen)
end

function julia_main()::Cint
    try
        s = ArgParseSettings(description="Syllable Detection")

        @add_arg_table s begin
            "fname"
            required = true
            arg_type = String
            help = "Input audio file."
            "output"
            required = true
            arg_type = String
            help = "Output CSV file."
            "--frame-length", "-l"
            arg_type = Int
            help = "Frame length in samples."
            default = 150
            "--overlap", "-o"
            arg_type = Int
            help = "Frame overlap in samples."
            default = 100
        end

        args = parse_args(s)
        audio = load(args["fname"])
        if !(typeof(audio) <: AbstractSampleBuf)
            data = audio[1]
            samplerate  = audio[3]
        else
            if nchannels(audio) == 2
                data = @. (audio.data[:, 1] * 0.5 + audio.data[:, 2] * 0.5)
            else
                data = audio.data
            end
            samplerate = audio.samplerate
        end
        starts, ends = getsyllables(data, samplerate, args["frame-length"], args["overlap"])
        if starts == ends == [0]
            println("No syllables detected.")
            return 0
        end
        CSV.write(args["output"], DataFrame(Start=starts, End=ends))
        println("Syllables CSV file generated: $(args["output"])")
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end
end
