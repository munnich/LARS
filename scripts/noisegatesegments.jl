using Statistics, CSV, FileIO, LibSndFile, SampledSignals, ArgParse, DataFrames

function noisegate(audio, noisesegment, minlength)
    noiselevel = maximum(noisesegment) * 2

    starts = Vector{Int}()
    stops = Vector{Int}()
    start = 0
    isnoise = true
    since = 0
    for i in 1:length(audio)
        if audio[i] > noiselevel
            since = 0
            if isnoise 
                start = i 
                isnoise = false
            end
        else
            if !isnoise
                since += 1
                if since > minlength && (i - since - start) > minlength
                    i -= since
                    push!(starts, start)
                    push!(stops, i)
                    since = 0
                    isnoise = true
                end
            end
        end
    end
    starts, stops
end

function main()
    s = ArgParseSettings(description="Segment Detection Using a Noise Gate")

    @add_arg_table s begin
        "fname"
        required = true
        arg_type = String
        help = "Input audio file."
        "noise-start"
        required = true
        arg_type = Int
        help = "Reference noise first sample."
        "noise-stop"
        required = true
        arg_type = Int
        help = "Reference noise last sample."
        "output"
        required = true
        arg_type = String
        help = "Output CSV file."
        "--min-length", "-m"
        required = false
        arg_type = Int
        help = "Minimum sample length for noise and non-noise"
        default = 4410
    end

    args = parse_args(s)
    audio = load(args["fname"])
    if !(typeof(audio) <: AbstractSampleBuf)
        audio = audio[1]
    else
        audio = audio.data
    end
    if size(audio, 2) == 2 
        audio = @. (audio[:, 1] + audio[:, 2]) / 2
    end
    audio = abs.(audio)
    noise = audio[(args["noise-start"]):(args["noise-stop"])]
    starts, stops = noisegate(audio, noise, args["min-length"])
    if starts == stops == [0]
        println("No segments detected.")
        return 0
    end
    CSV.write(args["output"], DataFrame(Start=starts, End=stops))
    println("Segments CSV file generated: $(args["output"])")
end

main()
